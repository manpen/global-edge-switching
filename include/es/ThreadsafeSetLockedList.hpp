#pragma once

#include <atomic>
#include <mutex>
#include <vector>

namespace es {

template<typename Value, typename HashFcn>
struct ThreadsafeSetLockedList {
public:
    ThreadsafeSetLockedList(size_t num_edges) : buckets_(num_edges), locks_(num_edges) {
        for (auto& lock : locks_) lock.store(true);
    }

    bool insert(Value val) {
        size_t key = hash_func_(val);
        size_t bucket = key % buckets_.size();
        lock(bucket);
        size_t bucket_size = buckets_[bucket].size();
        size_t pos = 0;
        while (pos < bucket_size) {
            if (buckets_[bucket][pos] == val) {
                unlock(bucket);
                return false;
            }
            pos++;
        }
        buckets_[bucket].push_back(val);
        unlock(bucket);
        return true;
    }

    bool erase(Value val) {
        size_t key = hash_func_(val);
        size_t bucket = key % buckets_.size();
        lock(bucket);
        size_t bucket_size = buckets_[bucket].size();
        for (size_t pos = 0; pos < bucket_size; ++pos) {
            if (buckets_[bucket][pos] == val) {
                buckets_[bucket][pos] = buckets_[bucket].back();
                buckets_[bucket].pop_back();
                unlock(bucket);
                return true;
            }
        }
        unlock(bucket);
        return false;
    }

private:
    void lock(size_t bucket) {
        bool available;
        do {
            available = true;
            locks_[bucket].compare_exchange_weak(available, false,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed);
        } while (!available);
    }

    void unlock(size_t bucket) {
        locks_[bucket].store(true, std::memory_order_release);
    }

    std::vector<std::vector<Value>> buckets_;
    std::vector<std::atomic<bool>> locks_;
    HashFcn hash_func_;
};

}