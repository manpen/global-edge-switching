#pragma once

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>

#include <tlx/math.hpp>

//#define EDGE_DEPS_STATS

namespace es {

template<typename HashFcn>
struct EdgeDependenciesNoWaitV3 {

    using switch_t = size_t;
    static_assert(sizeof(switch_t) >= sizeof(edge_t));

    static constexpr auto kEdgeBits = 8 * sizeof(edge_t);
    static constexpr auto kEdgeMask = edge_t(-1) >> 1;
    static constexpr auto kResolvedBit   = ~(switch_t(-1) >> 1);
    static constexpr auto kLastSwitch = kResolvedBit - 1;

    enum class DependencyType : edge_t {
        Erase = 0,
        Insert = 1
    };

    class EdgeDependency {
    public:
        EdgeDependency() {}

    // erase announcements
        void announce_erase(switch_t sid, std::memory_order order = std::memory_order_relaxed) {
            assert(from_key(edge_).second == DependencyType::Erase);
            switch_id_.store(sid, order);
        }

        void announce_erase_succeeded(switch_t sid, std::memory_order order = std::memory_order_release) {
            assert(from_key(edge_).second == DependencyType::Erase);
            assert(switch_id()  == sid);
            switch_id_.store(sid | kResolvedBit, order);
        }

        void announce_erase_failed(switch_t sid, std::memory_order order = std::memory_order_release) {
            assert(from_key(edge_).second == DependencyType::Erase);
            assert(switch_id()  == sid);
            switch_id_.store(kLastSwitch, order);
        }

    // insert announcements
        bool announce_insert_if_minimum(size_t sid) {
            assert(from_key(edge_).second == DependencyType::Insert);
            assert(sid < kResolvedBit);

            sid |= kResolvedBit;

            auto current_minimum = switch_id_.load(std::memory_order_consume);
            while(current_minimum > sid) {
                switch_id_.compare_exchange_weak(current_minimum, sid,
                                                      std::memory_order_release,
                                                      std::memory_order_consume);
            }

            return current_minimum <= sid;
        }

        void announce_insert_succeeded(switch_t sid, std::memory_order order = std::memory_order_release) {
            assert(from_key(edge_).second == DependencyType::Insert);
            assert(switch_id()  == sid);
            switch_id_.store(sid, order);
        }

        void announce_insert_failed(switch_t sid, std::memory_order order = std::memory_order_release) {
            assert(from_key(edge_).second == DependencyType::Insert);
            assert(switch_id()  == sid);
            switch_id_.store(kLastSwitch | kResolvedBit, order);
        }

    // wrappers
        [[nodiscard]] constexpr edge_t edge(std::memory_order order = std::memory_order_consume) const {
            return from_key(edge_.load(order)).first;
        }

        [[nodiscard]] constexpr DependencyType dependency_type(std::memory_order order = std::memory_order_consume) const {
            return from_key(edge_.load(order)).second;
        }

        [[nodiscard]] constexpr switch_t switch_id(std::memory_order order = std::memory_order_consume) const {
            auto tmp = switch_id_.load(order);
            return tmp & ~kResolvedBit;
        }

        [[nodiscard]] constexpr std::pair<switch_t, bool> get_erase_switch_id_and_resolved(std::memory_order order = std::memory_order_consume) const {
            auto tmp = switch_id_.load(order);
            return {tmp & ~kResolvedBit, tmp & kResolvedBit};
        }

        [[nodiscard]] constexpr std::pair<switch_t, bool> get_insert_switch_id_and_resolved(std::memory_order order = std::memory_order_consume) const {
            auto tmp = switch_id_.load(order);
            return {tmp & ~kResolvedBit, !(tmp & kResolvedBit)};
        }

    private:
        std::atomic<edge_t> edge_ = std::numeric_limits<size_t>::max();
        std::atomic<switch_t> switch_id_ = std::numeric_limits<switch_t>::max();
        friend EdgeDependenciesNoWaitV3;
    };

    const edge_t kEmpty_ = std::numeric_limits<edge_t>::max();

    using iterator_t = EdgeDependency*;

    EdgeDependenciesNoWaitV3(size_t num_edges, double load_factor)
        : dependencies_(tlx::round_up_to_power_of_two(static_cast<size_t>(load_factor * 2 * num_edges + 1)))
        , mod_mask_(dependencies_.size() - 1)
    {
        clear();
    }

#ifdef EDGE_DEPS_STATS
    ~EdgeDependenciesNoWaitV3() {
        std::cout << "Iters/Calls: " << (1.0 * stat_iters / stat_calls) << std::endl;
        auto num_empty = std::count_if(dependencies_.begin(), dependencies_.end(), [&] (auto & entry) {return entry.edge_ == kEmpty_;});
        std::cout << "Load factor: " << (1.0 - 1.0 * num_empty / dependencies_.size()) << std::endl;
    }
#endif

    void next_round() {
        clear();
    }

    void clear() {
        #pragma omp parallel for schedule(dynamic, 1024)
        for(size_t i = 0; i < dependencies_.size(); ++i) {
            dependencies_[i].edge_.store(kEmpty_, std::memory_order_relaxed);
            dependencies_[i].switch_id_.store(edge_t(-1), std::memory_order_relaxed);
        }
    }

    using hint_t = std::pair<edge_t, size_t>;
    hint_t prefetch(edge_t edge, DependencyType dep_type) const {
        const auto key = to_key(edge, dep_type);
        size_t bucket = hash_func_(edge) & mod_mask_;
        __builtin_prefetch(dependencies_.data() + bucket, 1, 1);
        __builtin_prefetch(dependencies_.data() + bucket + 1, 1, 1);
        return {key, bucket};
    }

    std::pair<iterator_t, bool> find_or_insert(edge_t edge, DependencyType dep_type) {
        return find_or_insert_<true>(edge, dep_type);
    }

    [[nodiscard]] iterator_t find(edge_t edge, DependencyType dep_type) {
        return find_or_insert_<false>(edge, dep_type).first;
    }

    std::pair<iterator_t, bool> find_or_insert(hint_t hint) {
        return find_or_insert_<true>(hint);
    }

    [[nodiscard]] iterator_t find(hint_t hint) {
        return find_or_insert_<false>(hint).first;
    }

    iterator_t announce_erase(edge_t edge, switch_t sid) {
        auto [iter, is_new] = find_or_insert_<true>(edge, DependencyType::Erase);
        assert(is_new);
        iter->announce_erase(sid);
        return iter;
    }

    iterator_t announce_erase(hint_t hint, switch_t sid) {
        auto [iter, is_new] = find_or_insert_<true>(hint);
        assert(is_new);
        iter->announce_erase(sid);
        return iter;
    }

    iterator_t announce_insert_if_minimum(edge_t edge, switch_t sid) {
        auto iter = find_or_insert_<true>(edge, DependencyType::Insert).first;
        iter->announce_insert_if_minimum(sid);
        return iter;
    }

    iterator_t announce_insert_if_minimum(hint_t hint, switch_t sid) {
        auto iter = find_or_insert_<true>(hint).first;
        iter->announce_insert_if_minimum(sid);
        return iter;
    }

private:
    std::vector<EdgeDependency> dependencies_;
    HashFcn hash_func_;
    size_t mod_mask_;

#ifdef EDGE_DEPS_STATS
    std::atomic<size_t> stat_calls{0};
    std::atomic<size_t> stat_iters{0};
#endif

    [[nodiscard]] static constexpr edge_t to_key(edge_t edge, DependencyType type) {
        assert(edge <= kEdgeMask);
        return edge | (static_cast<edge_t>(type) << (kEdgeBits - 1));
    }

    [[nodiscard]] static constexpr std::pair<edge_t, DependencyType> from_key(edge_t edge) {
        return {edge & kEdgeMask, static_cast<DependencyType>(edge >> (kEdgeBits - 1))};
    }


    template <bool AllowInsert>
    [[nodiscard]] std::pair<iterator_t, bool> find_or_insert_(edge_t edge, DependencyType dep_type) {
        const auto key = to_key(edge, dep_type);
        size_t bucket = hash_func_(edge) & mod_mask_;
        return find_or_insert_<AllowInsert>({key, bucket});
    }

    template <bool AllowInsert>
    [[nodiscard]] std::pair<iterator_t, bool> find_or_insert_(hint_t hint) {
        auto [key,  bucket] = hint;

#ifdef EDGE_DEPS_STATS
        ++stat_calls;
#endif

        while (true) {
#ifdef EDGE_DEPS_STATS
            ++stat_iters;
#endif
            auto iter = dependencies_.data() + bucket;

            if (AllowInsert) {
                auto test_empty_value = kEmpty_;
                auto did_exchange =
                iter->edge_.compare_exchange_strong(test_empty_value, key,
                                                   std::memory_order_release,
                                                   std::memory_order_consume);

                if (did_exchange || test_empty_value == key)
                    return {iter, did_exchange};
            } else {
                const auto value = iter->edge_.load(std::memory_order_consume);

                if (value == key)
                    return {iter, false};

                if (value == kEmpty_)
                    return {nullptr, false};
            }

            bucket = (bucket + 1) & mod_mask_;
        }
    }

};

}