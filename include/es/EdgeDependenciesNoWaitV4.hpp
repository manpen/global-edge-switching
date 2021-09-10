#pragma once

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>

#include <tlx/math.hpp>

//#define USE_FASTRANGE
#ifdef USE_FASTRANGE
#include <fastrange.h>
#endif

//#define EDGE_DEPS_STATS

namespace es {

template<typename HashFcn>
struct EdgeDependenciesNoWaitV4 {

    using switch_t = size_t;
    static_assert(sizeof(switch_t) >= sizeof(edge_t));

    static constexpr auto kRoundBits = 8;
    static constexpr auto kEdgeMask = edge_t(-1) >> kRoundBits;
    static constexpr auto kRoundShift = (sizeof(edge_t) * 8) - kRoundBits;
    static constexpr auto kLocked = edge_t(-1) >> kRoundShift;

    static constexpr auto kResolvedBit   = ~(switch_t(-1) >> 1);
    static constexpr auto kSwitchMask = kResolvedBit - 1;
    static constexpr auto kLastSwitch = kSwitchMask;

    static constexpr auto kEraseInit = kLastSwitch - 1;
    static constexpr auto kInsertInit = switch_t(-1);

    class EdgeDependency {
    public:
        EdgeDependency() {}

    // erase announcements
        void announce_erase(switch_t sid, std::memory_order order = std::memory_order_relaxed) {
            assert(erase_switch_id_ == kEraseInit);
            erase_switch_id_.store(sid, order);
        }

        void announce_erase_succeeded(switch_t sid, std::memory_order order = std::memory_order_release) {
            assert((erase_switch_id_ & kSwitchMask) == sid);
            erase_switch_id_.store(sid | kResolvedBit, order);
        }

        void announce_erase_failed(switch_t sid, std::memory_order order = std::memory_order_release) {
            assert((erase_switch_id_ & kSwitchMask) == sid);
            erase_switch_id_.store(kLastSwitch, order);
        }

    // insert announcements
        bool announce_insert_if_minimum(size_t sid) {
            assert(sid < kResolvedBit);

            sid |= kResolvedBit;

            auto current_minimum = insert_switch_id_.load(std::memory_order_consume);
            while(current_minimum > sid) {
                insert_switch_id_.compare_exchange_weak(current_minimum, sid,
                                                      std::memory_order_release,
                                                      std::memory_order_consume);
            }

            return current_minimum <= sid;
        }

        void announce_insert_succeeded(switch_t sid, std::memory_order order = std::memory_order_release) {
            assert((insert_switch_id_ & kSwitchMask) == sid);
            insert_switch_id_.store(sid, order);
        }

        void announce_insert_failed(switch_t sid, std::memory_order order = std::memory_order_release) {
            assert((insert_switch_id_ & kSwitchMask) == sid);
            insert_switch_id_.store(kLastSwitch | kResolvedBit, order);
        }

    // wrappers
        [[nodiscard]] constexpr edge_t edge(std::memory_order order = std::memory_order_consume) const {
            return edge_.load(order) & kEdgeMask;
        }

        [[nodiscard]] constexpr std::tuple<bool, switch_t, bool> get_erase_switch_id_and_resolved(std::memory_order order = std::memory_order_consume) const {
            auto tmp = erase_switch_id_.load(order);
            return {tmp == kEraseInit, tmp & ~kResolvedBit, tmp & kResolvedBit};
        }

        [[nodiscard]] constexpr std::pair<switch_t, bool> get_insert_switch_id_and_resolved(std::memory_order order = std::memory_order_consume) const {
            auto tmp = insert_switch_id_.load(order);
            return {tmp & ~kResolvedBit, !(tmp & kResolvedBit)};
        }

    private:
        std::atomic<edge_t> edge_;
        std::atomic<switch_t> erase_switch_id_;
        std::atomic<switch_t> insert_switch_id_;
        friend EdgeDependenciesNoWaitV4;
    };


    using iterator_t = EdgeDependency*;

    EdgeDependenciesNoWaitV4(size_t num_edges, double load_factor)
#ifdef USE_FASTRANGE
    : dependencies_(static_cast<size_t>(load_factor * 2 * num_edges + 1))
#else
        : dependencies_(tlx::round_up_to_power_of_two(static_cast<size_t>(load_factor * 2 * num_edges + 1)))
        , num_edges_(num_edges)
        , mod_mask_(dependencies_.size() - 1)
#endif
    {
        clear();
    }

#if 0
    ~EdgeDependenciesNoWaitV4() {
#ifdef EDGE_DEPS_STATS
        std::cout << "Iters/Calls: " << (1.0 * stat_iters / stat_calls) << std::endl;
#endif
        auto size = std::count_if(dependencies_.begin(), dependencies_.end(), [&] (auto & entry) {return from_key(entry.edge_).second == round_;});
        std::cout << "Size: " << size << "\n";
        std::cout << "Load factor: " << (1.0 * size / dependencies_.size()) << "\n";
        std::cout << "Entries per switch: " << 1.0 * size / (num_edges_ / 2) << std::endl;
    }
#endif

    void next_round() {
        if (++round_ == kLocked) {
            clear();
        }
    }

    void clear() {
        auto n = dependencies_.size();

        #pragma omp parallel for schedule(dynamic, 1024)
        for(size_t i = 0; i < n; ++i) {
            dependencies_[i].edge_.store(0, std::memory_order_relaxed);
            dependencies_[i].erase_switch_id_.store(kEraseInit, std::memory_order_relaxed);
            dependencies_[i].insert_switch_id_.store(kInsertInit, std::memory_order_relaxed);
        }

        round_ = 1;
    }


    using hint_t = std::pair<edge_t, size_t>;
    hint_t prefetch(edge_t edge) const {
        size_t bucket = first_bucket_of_hash(hash_func_(edge));
        __builtin_prefetch(dependencies_.data() + bucket, 1, 1);
        __builtin_prefetch(dependencies_.data() + bucket + 1, 1, 1);
        return {edge, bucket};
    }

    std::pair<iterator_t, bool> find_or_insert(edge_t edge) {
        return find_or_insert_<true>(edge);
    }

    [[nodiscard]] iterator_t find(edge_t edge) {
        return find_or_insert_<false>(edge).first;
    }

    std::pair<iterator_t, bool> find_or_insert(hint_t hint) {
        return find_or_insert_<true>(hint);
    }

    [[nodiscard]] iterator_t find(hint_t hint) {
        return find_or_insert_<false>(hint).first;
    }

    iterator_t announce_erase(edge_t edge, switch_t sid) {
        auto [iter, is_new] = find_or_insert_<true>(edge);
        assert(is_new);
        iter->announce_erase(sid);
        return iter;
    }

    iterator_t announce_erase(hint_t hint, switch_t sid) {
        auto [iter, is_new] = find_or_insert_<true>(hint);
        iter->announce_erase(sid);
        return iter;
    }

    iterator_t announce_insert_if_minimum(edge_t edge, switch_t sid) {
        auto iter = find_or_insert_<true>(edge).first;
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
    size_t num_edges_;
    unsigned round_ = 1;

#ifdef EDGE_DEPS_STATS
    std::atomic<size_t> stat_calls{0};
    std::atomic<size_t> stat_iters{0};
#endif

    template <bool AllowInsert>
    [[nodiscard]] std::pair<iterator_t, bool> find_or_insert_(edge_t edge) {
        assert(edge >> kRoundShift == 0);
        size_t bucket = first_bucket_of_hash(hash_func_(edge));
        return find_or_insert_<AllowInsert>({edge, bucket});
    }

    template <bool AllowInsert>
    [[nodiscard]] std::pair<iterator_t, bool> find_or_insert_(hint_t hint) {
        auto [edge,  bucket] = hint;

        auto key = to_key(edge, round_);
        auto locked_key = to_key(edge, kLocked);

#ifdef EDGE_DEPS_STATS
        ++stat_calls;
#endif

        while (true) {
#ifdef EDGE_DEPS_STATS
            ++stat_iters;
#endif
            auto iter = dependencies_.data() + bucket;
            auto current_key = iter->edge_.load(std::memory_order_consume);

            if (current_key == key)
                return {iter, false};

            // check whether cell is empty (i.e. from a previous round)
            if (from_key(current_key).second != round_) {
                if (TLX_UNLIKELY(is_locked(current_key)))
                    continue;

                if (AllowInsert) {
                    constexpr auto order = std::memory_order_release;

                    auto did_exchange = iter->edge_.compare_exchange_strong(current_key, locked_key, order,
                                                                          std::memory_order_consume);

                    if (TLX_UNLIKELY(!did_exchange))
                        continue; // try again

                    iter->erase_switch_id_.store(kEraseInit, order);
                    iter->insert_switch_id_.store(kInsertInit, order);
                    iter->edge_.store(key, order);

                    return {iter, true};
                 } else {
                    return {nullptr, false};
                }
            }

            bucket = next_cyclic_bucket(bucket);
        }
    }
#ifdef USE_FASTRANGE
    [[nodiscard]] size_t first_bucket_of_hash(size_t hash) const {
        return sizeof(size_t) == 8 ? fastrange64(hash, dependencies_.size()) : fastrange32(hash, dependencies_.size());
    }

    [[nodiscard]] size_t next_cyclic_bucket(size_t bucket) const {
        ++bucket;
        if (bucket == dependencies_.size()) bucket = 0;
        return bucket;
    }

#else
    [[nodiscard]] size_t first_bucket_of_hash(size_t hash) const {
        return (hash ^ (hash >> 32)) & mod_mask_;
    }

    [[nodiscard]] size_t next_cyclic_bucket(size_t bucket) const {
        return (bucket + 1) & mod_mask_;
    }
#endif

    constexpr static edge_t to_key(edge_t edge, unsigned round) {
        assert((edge >> kRoundShift) == 0);
        assert((round >> kRoundBits) == 0);
        return edge | (edge_t(round) << kRoundShift);
    }

    constexpr static std::pair<edge_t, unsigned> from_key(edge_t key) {
        return {key & kEdgeMask, key >> kRoundShift};
    }

    constexpr static bool is_locked(edge_t key) {
        return from_key(key).second == kLocked;
    }


};

}