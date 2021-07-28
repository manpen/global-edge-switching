#pragma once

#include <atomic>
#include <mutex>
#include <vector>

namespace es {

template<typename Value, typename HashFcn>
struct ThreadsafeSetLockedList {
public:
    ThreadsafeSetLockedList(size_t num_edges) : buckets_(num_edges), locks_(num_edges), num_edges_(num_edges) {}

    bool insert(Value val) {
        size_t key = hash_func_(val);
        size_t bucket = key % buckets_.size();
        std::lock_guard<std::mutex> lock(locks_[bucket]);
        size_t bucket_size = buckets_[bucket].size();
        size_t pos = 0;
        while (pos < bucket_size) {
            if (buckets_[bucket][pos] == val) return false;
            pos++;
        }
        buckets_[bucket].push_back(val);
        return true;
    }

    bool erase(Value val) {
        size_t key = hash_func_(val);
        size_t bucket = key % buckets_.size();
        std::lock_guard<std::mutex> lock(locks_[bucket]);
        size_t bucket_size = buckets_[bucket].size();
        for (size_t pos = 0; pos < bucket_size; ++pos) {
            if (buckets_[bucket][pos] == val) {
                buckets_[bucket][pos] = buckets_[bucket].back();
                buckets_[bucket].pop_back();
                return true;
            }
        }
        return false;
    }

private:
    std::vector<std::vector<Value>> buckets_;
    std::vector<std::mutex> locks_;
    HashFcn hash_func_;
    size_t num_edges_;
};

}