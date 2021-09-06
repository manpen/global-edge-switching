#pragma once

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>

#include <tlx/math.hpp>

namespace es {

template<typename HashFcn>
struct EdgeDependenciesNoWait {
public:
    const size_t kNone_ = std::numeric_limits<size_t>::max();
    const edge_t kEmpty_ = std::numeric_limits<edge_t>::max();
    const int kLocked_ = std::numeric_limits<int>::max();

    EdgeDependenciesNoWait(size_t num_edges, double load_factor)
        : dependencies_(tlx::round_up_to_power_of_two(static_cast<size_t>(load_factor * 2 * num_edges + 1)))
        , mod_mask_(dependencies_.size() - 1)
        , round_(0) {}

    void next_round() {
        round_++;
    }

    void announce_erase(edge_t edge, size_t sid) {
        size_t bucket = hash_func_(edge) & mod_mask_;
        while (true) {
            auto iter = dependencies_.begin() + bucket;
            int round_at_iter = iter->round.load(std::memory_order_acquire);
            if (round_at_iter < round_) {
                iter->round.compare_exchange_strong(round_at_iter, round_,
                                                    std::memory_order_release,
                                                    std::memory_order_consume);
                if (round_at_iter < round_) {
                    iter->edge = edge;
                    iter->type = ERASE;
                    iter->switch_id = sid;
                    iter->resolved = false;
                    break;
                }
            }
            bucket = (bucket + 1) & mod_mask_;
        }
    }

    bool announce_insert_if_minimum(edge_t edge, size_t sid) {
        size_t bucket = hash_func_(edge) & mod_mask_;
        while (true) {
            auto iter = dependencies_.begin() + bucket;
            int round_at_iter = iter->round.load(std::memory_order_acquire);
            if (round_at_iter < round_) {
                iter->round.compare_exchange_strong(round_at_iter, kLocked_,
                                                    std::memory_order_release,
                                                    std::memory_order_consume);
                if (round_at_iter < round_) {
                    iter->edge = edge;
                    iter->type = INSERT;
                    iter->switch_id = sid;
                    iter->resolved = false;
                    iter->round = round_;
                    return true;
                }
                continue;
            }
            if (round_at_iter == kLocked_) continue;
            if (iter->edge == edge && iter->type == INSERT) {
                size_t switch_at_iter = iter->switch_id.load(std::memory_order_acquire);
                while (switch_at_iter > sid) {
                    iter->switch_id.compare_exchange_weak(switch_at_iter, sid,
                                                          std::memory_order_release,
                                                          std::memory_order_consume);
                }
                return switch_at_iter == sid;
            }
            bucket = (bucket + 1) & mod_mask_;
        }
    }

    void announce_erase_succeeded(edge_t edge, size_t sid) {
        auto iter = find_erase(edge, sid);
        iter->resolved = true;
    }

    void announce_erase_failed(edge_t edge, size_t sid) {
        auto iter = find_erase(edge, sid);
        iter->switch_id = kNone_;
    }

    void announce_insert_succeeded(edge_t edge, size_t sid) {
        auto iter = find_insert(edge, sid);
        iter->resolved = true;
    }

    void announce_insert_failed(edge_t edge, size_t sid) {
        auto iter = find_insert(edge, sid);
        iter->switch_id = kNone_;
    }

    [[nodiscard]] std::tuple<size_t, bool> lookup_erase(edge_t edge) const {
        size_t bucket = hash_func_(edge) & mod_mask_;
        while (true) {
            auto iter = dependencies_.begin() + bucket;
            if (iter->round < round_) return {0, true};
            if (iter->edge == edge && iter->type == ERASE) return {iter->switch_id, iter->resolved};
            bucket = (bucket + 1) & mod_mask_;
        }
    }

    [[nodiscard]] std::tuple<size_t, bool> lookup_insert(edge_t edge) const {
        size_t bucket = hash_func_(edge) & mod_mask_;
        while (true) {
            auto iter = dependencies_.begin() + bucket;
            if (iter->round < round_) return {kNone_, false};
            if (iter->edge == edge && iter->type == INSERT) return {iter->switch_id, iter->resolved};
            bucket = (bucket + 1) & mod_mask_;
        }
    }

private:
    enum DependencyType : char {
        NONE = 0,
        ERASE = 1,
        INSERT = 2
    };

    struct EdgeDependency {
        edge_t edge = std::numeric_limits<size_t>::max();
        std::atomic<size_t> switch_id = 0;
        std::atomic<int> round = -1;
        bool resolved = true;
        DependencyType type = NONE;
    };

    std::vector<EdgeDependency> dependencies_;
    HashFcn hash_func_;
    int round_;
    size_t mod_mask_;

    using iterator_t = typename std::vector<EdgeDependency>::iterator;

    iterator_t find_erase(edge_t edge, size_t sid) {
        size_t bucket = hash_func_(edge) & mod_mask_;
        while (true) {
            auto iter = dependencies_.begin() + bucket;
            if (iter->edge == edge && iter->type == ERASE && iter->switch_id == sid) return iter;
            bucket = (bucket + 1) & mod_mask_;
        }
    }

    iterator_t find_insert(edge_t edge, size_t sid) {
        size_t bucket = hash_func_(edge) & mod_mask_;
        while (true) {
            auto iter = dependencies_.begin() + bucket;
            if (iter->edge == edge && iter->type == INSERT && iter->switch_id == sid) return iter;
            bucket = (bucket + 1) & mod_mask_;
        }
    }
};

}