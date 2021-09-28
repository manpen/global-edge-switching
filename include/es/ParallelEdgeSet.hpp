#pragma once

#include <cassert>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <vector>
#include <memory>

#include <omp.h>

#include <nmmintrin.h>

#include <tlx/define/likely.hpp>
#include <tlx/math.hpp>

#include <tsl/robin_growth_policy.h>

#include <es/Graph.hpp>
#include <es/ScopedTimer.hpp>

template<typename Growth = tsl::rh::power_of_two_growth_policy<2>>
class ParallelEdgeSet {
public:
#if 0
    using value_type = unsigned __int128;
    using node_type = uint64_t;
    using iterator_type = value_type *;
    static constexpr size_t kBitsPerNode = 48;
#else
    using value_type = uint64_t;
    using node_type = uint32_t;
    using iterator_type = value_type *;
    static constexpr size_t kBitsPerNode = 28;
#endif

    using hint_t = iterator_type;

    static constexpr size_t kBitsPerPayload = 2 * kBitsPerNode;
    static constexpr node_type kNodeMask = (node_type(1) << kBitsPerNode) - 1;

    static constexpr value_type kEmpty = -1;
    static constexpr value_type kDeleted = kEmpty - 1;

    ParallelEdgeSet(size_t max_size, double load_factor = 4.0, uint64_t hash_bias = 0x12345678) : max_size_(
        static_cast<size_t>(max_size * load_factor)), growth_(max_size_), storage_(max_size_ + 1, kEmpty), hash_bias_(hash_bias) {
        assert(load_factor >= 1.0);

        // align pointer
        data_begin_ = reinterpret_cast<iterator_type>(
            tlx::div_ceil<intptr_t>(reinterpret_cast<intptr_t>(storage_.data()), alignof(value_type)) * alignof(value_type));

        data_end_ = data_begin_ + max_size_;
    }

    ParallelEdgeSet(ParallelEdgeSet &&) = default;

    ParallelEdgeSet &operator=(ParallelEdgeSet &&) = default;

    template <bool DoPrefetch = true>
    hint_t prefetch(node_type a, node_type b) const noexcept {
        auto reference_payload = build_bucket(a, b, 0);
        auto bucket = growth_.bucket_for_hash(hash(reference_payload));
        auto it = data_begin_ + bucket;

        if (DoPrefetch)
            __builtin_prefetch(it, 1, 1);

        return it;
    }

    [[nodiscard]] size_t bucket_of(iterator_type it) const noexcept {
        assert(data_begin_ <= it);
        assert(it <= data_end_);
        return static_cast<size_t>(std::distance(data_begin_, it));
    }

    void prefetch_bucket(size_t bucket) const noexcept {
        auto it = data_begin_ + bucket;
        __builtin_prefetch(it, 1, 1);
    }

    [[nodiscard]] iterator_type lock_bucket(size_t bucket, unsigned tid) noexcept {
        assert(bucket < std::distance(data_begin_, data_end_));
        auto it = data_begin_ + bucket;
        while(true) {
            auto value = __atomic_load_n(it, __ATOMIC_CONSUME);
            if (value == kEmpty || value == kDeleted || is_locked(value)) return nullptr;

            auto locked = build_bucket(value, tid);
            auto did_acquire = __atomic_compare_exchange_n(it, &value, locked, false, __ATOMIC_RELEASE, __ATOMIC_CONSUME);

            if (did_acquire)
                return it;
        }
    }

    [[nodiscard]] std::pair<bool, iterator_type> acquire(node_type a, node_type b, unsigned tid) noexcept {
        auto reference_payload = build_bucket(a, b, 0);
        auto bucket = growth_.bucket_for_hash(hash(reference_payload));
        auto it = data_begin_ + bucket;
        return acquire(it, a, b, tid);
    }

    // returns pair {found, ticket} --- the first entry is true iff the item is in the set;
    // the second entry returns a pointer if the bucket could be locked
    [[nodiscard]] std::pair<bool, iterator_type> acquire(hint_t it, node_type a, node_type b, unsigned tid) noexcept {
        auto reference_unlocked = build_bucket(a, b);
        auto reference_locked   = build_bucket(a, b, tid);
        auto reference_payload  = get_payload(reference_unlocked);

        while (true) {
            auto value = reference_unlocked;
            auto did_acquire = __atomic_compare_exchange_n(it, &value, reference_locked, false, __ATOMIC_RELEASE,
                                                           __ATOMIC_CONSUME);
            if (did_acquire)
                return {true, it};

            if (value == kEmpty)
                return {false, nullptr};

            if (get_payload(value) == reference_payload) {
                return {true, nullptr};
            }

            ++it;
            if (TLX_UNLIKELY(it == data_end_))
                it = data_begin_;
        }
    }

    // if successful returns a ticket to the locked bucket; if item already exists return nullptr
    [[nodiscard]] iterator_type insert(node_type a, node_type b, unsigned tid = -1) noexcept {
        const auto reference_locked = build_bucket(a, b, tid);
        return insert(reference_locked);
    }

    [[nodiscard]] iterator_type insert(hint_t hint, node_type a, node_type b, unsigned tid = -1) noexcept {
        const auto reference_locked = build_bucket(a, b, tid);
        const auto reference_payload = get_payload(reference_locked);

        return insert(hint, reference_payload, reference_locked);
    }

    void release(iterator_type it) {
        assert(data_begin_ <= it);
        assert(it < data_end_);
        assert(is_locked(*it));

        // TODO: we assume that the iterator was acquired from this thread, i.e., we may read from it non-atomically ?!
        auto tmp = *it | (kEmpty << kBitsPerPayload); // remove thread-lock
        __atomic_store_n(it, tmp, __ATOMIC_RELEASE);
    }

    void erase_and_release(iterator_type it) {
        assert(data_begin_ <= it);
        assert(it < data_end_);
        assert(is_locked(*it));

        __atomic_store_n(it, kDeleted, __ATOMIC_RELEASE);
    }

    void erase(node_type a, node_type b) {
        auto reference_unlocked = build_bucket(a, b);
        auto reference_payload  = get_payload(reference_unlocked);

        auto bucket = growth_.bucket_for_hash(hash(reference_payload));
        auto it = data_begin_ + bucket;

        while (true) {
            auto value = reference_unlocked;
            auto did_erase = __atomic_compare_exchange_n(it, &value, kDeleted, false,
                                                         __ATOMIC_RELEASE,
                                                         __ATOMIC_CONSUME);
            if (did_erase)
                return;

            assert(value != kEmpty);

            if (get_payload(value) == reference_payload) continue;

            ++it;
            if (TLX_UNLIKELY(it == data_end_))
                it = data_begin_;
        }
    }


    // rebuilding
    void rebuild() {
        auto size = storage_.size() - 1;
        bool fresh_rebuild_instance = !rebuild_instance_;
        if (fresh_rebuild_instance) {
            rebuild_instance_ = std::make_unique<ParallelEdgeSet<Growth>>(size, 1.0, hash_bias_);
        }

        size_t empty_slots = 0;
        size_t num_items = 0;

        #pragma omp parallel reduction(+:empty_slots, num_items)
        {
            if (!fresh_rebuild_instance) {
                // parallel fill(rebuild_instance->data_begin, end, kEmpty);
                auto data = rebuild_instance_->data_begin_;
                #pragma omp for schedule(dynamic,100000)
                for (size_t i = 0; i != size; ++i) {
                    data[i] = kEmpty;
                }
            }

            #pragma omp for schedule(dynamic,100000)
            for (size_t i = 0; i != size; ++i) {
                auto value = data_begin_[i];
                empty_slots += (value == kEmpty);
                if (value == kEmpty || value == kDeleted)
                    continue;

                assert(!is_locked(value));

                auto fast_insert = [] (auto& inst, auto value)
                {
                    auto bucket = inst.growth_.bucket_for_hash(inst.hash(inst.get_payload(value)));
                    auto it = inst.data_begin_ + bucket;

                    while (true) {
                        auto empty = kEmpty;
                        if (__atomic_compare_exchange_n(it, &empty, value, false, __ATOMIC_RELEASE, __ATOMIC_CONSUME))
                            break;

                        ++it;
                        if (TLX_UNLIKELY(it == inst.data_end_))
                            it = inst.data_begin_;
                    }
                };
                fast_insert(*rebuild_instance_, value);

                ++num_items;
            }
        }

        // we will now switch the rebuild instance with us; only one thing we have to make sure:
        // we do not want to switch the pointer to the rebuild instance (next time, we will rebuild
        // it will start again from *this and not from *rebuild_instance).
        auto ptr_to_org_rebuild_instance = rebuild_instance_.get();
        std::swap(*this, *rebuild_instance_);
        rebuild_instance_ = std::move(ptr_to_org_rebuild_instance->rebuild_instance_);
    }

    // rebuilding
    void rebuild(std::atomic<size_t>* non_empty) {
        auto size = storage_.size() - 1;
        bool fresh_rebuild_instance = !rebuild_instance_;
        if (fresh_rebuild_instance) {
            rebuild_instance_ = std::make_unique<ParallelEdgeSet<Growth>>(size, 1.0, hash_bias_);
        }

        size_t empty_slots = 0;
        size_t num_items = 0;

        std::vector<size_t> local_counts(omp_get_max_threads());
        std::vector<size_t> partial_sums(omp_get_max_threads());

        constexpr size_t kChuckSize = 1 << 14;

        #pragma omp parallel reduction(+:empty_slots, num_items)
        {
            const auto tid = static_cast<unsigned>(omp_get_thread_num());

            if (!fresh_rebuild_instance) {
                // parallel fill(rebuild_instance->data_begin, end, kEmpty);
                auto data = rebuild_instance_->data_begin_;
                #pragma omp for schedule(dynamic,kChuckSize)
                for (size_t i = 0; i != size; ++i) {
                    data[i] = kEmpty;
                }
            }

            size_t local_count = 0;
            #pragma omp for schedule(static, kChuckSize)
            for (size_t i = 0; i != size; ++i) {
                auto value = data_begin_[i];
                empty_slots += (value == kEmpty);
                local_count += !(value == kEmpty || value == kDeleted);
            }

            #pragma omp critical
            local_counts[tid] = local_count;

            #pragma omp barrier

            #pragma omp single // implies barrier at end
            std::partial_sum(begin(local_counts), end(local_counts), begin(partial_sums));

            auto my_writer = non_empty + (partial_sums[tid] - local_count);

            #pragma omp for schedule(static, kChuckSize)
            for (size_t i = 0; i != size; ++i) {
                auto value = data_begin_[i];
                if (value == kEmpty || value == kDeleted)
                    continue;

                assert(!is_locked(value));

                auto fast_insert = [] (auto& inst, auto value)
                {
                    auto bucket = inst.growth_.bucket_for_hash(inst.hash(inst.get_payload(value)));
                    auto it = inst.data_begin_ + bucket;

                    while (true) {
                        auto empty = kEmpty;
                        if (__atomic_compare_exchange_n(it, &empty, value, false, __ATOMIC_RELEASE, __ATOMIC_CONSUME)) {
                            return it;
                        }

                        ++it;
                        if (TLX_UNLIKELY(it == inst.data_end_))
                            it = inst.data_begin_;
                    }
                };
                *(my_writer++) = rebuild_instance_->bucket_of(fast_insert(*rebuild_instance_, value));

                ++num_items;
            }
        }

        // we will now switch the rebuild instance with us; only one thing we have to make sure:
        // we do not want to switch the pointer to the rebuild instance (next time, we will rebuild
        // it will start again from *this and not from *rebuild_instance).
        auto ptr_to_org_rebuild_instance = rebuild_instance_.get();
        std::swap(*this, *rebuild_instance_);
        rebuild_instance_ = std::move(ptr_to_org_rebuild_instance->rebuild_instance_);
    }


    [[nodiscard]] size_t capacity() const noexcept {
        return max_size_;
    }

    [[nodiscard]] std::tuple<node_type, node_type, value_type, bool> fetch(size_t bucket) const noexcept {
        return fetch(data_begin_ + bucket);
    }

    [[nodiscard]] std::tuple<node_type, node_type, value_type, bool> fetch(iterator_type it) const noexcept {
        auto edge = __atomic_load_n(it, __ATOMIC_CONSUME);
        auto payload = get_payload(edge);
        return {payload & kNodeMask, (payload >> kBitsPerNode), payload, payload == kEmpty || payload == kDeleted};
    }

private:
    size_t max_size_;
    Growth growth_;

    uint64_t hash_bias_;

    std::vector<value_type> storage_;

    value_type *data_begin_;
    value_type *data_end_;

    std::unique_ptr<ParallelEdgeSet> rebuild_instance_;

    template <bool EnsureUnique = true>
    iterator_type insert(value_type reference_locked) noexcept {
        const auto reference_payload = get_payload(reference_locked);

        auto bucket = growth_.bucket_for_hash(hash(reference_payload));
        auto it = data_begin_ + bucket;

        return insert<EnsureUnique>(it, reference_payload, reference_locked);
    }

    template <bool EnsureUnique = true>
    iterator_type insert(iterator_type it, value_type reference_payload, value_type reference_locked) noexcept {
        constexpr auto first_try = kEmpty;
        constexpr auto second_try = (first_try == kEmpty) ? kDeleted : kEmpty;
        value_type value_at_it = first_try;
        while (true) {
            if (__atomic_compare_exchange_n(it, &value_at_it, reference_locked, true, __ATOMIC_RELEASE, __ATOMIC_CONSUME)) {
                // we now have our foot in the door: we have a ticket, but still have to make sure
                // that the value is not yet stored in the hash set -- so we search on; if we find
                // it, we delete our temporary ticket, otherwise we make it permanent

                if (!EnsureUnique)
                    return it;

                auto search_it = it;
                while (true) {
                    ++search_it;
                    if (TLX_UNLIKELY(search_it == data_end_)) {
                        search_it = data_begin_;
                    }

                    auto search_value = __atomic_load_n(search_it, __ATOMIC_CONSUME);

                    if (search_value == kEmpty)
                        return it; // successful

                    if (get_payload(search_value) == reference_payload) {
                        // we found a bucket with the same payload
                        if (TLX_UNLIKELY(search_it == it)) {
                            // wrapped around, so we have the only ticket
                            return it;
                        }

                        // value still existed; remove ticket
                        __atomic_store_n(it, kDeleted, __ATOMIC_RELEASE);
                        return nullptr;
                    }
                }
            }

            if (value_at_it == second_try)
                continue;

            if (get_payload(value_at_it) == reference_payload)
                return nullptr;

            ++it;
            if (TLX_UNLIKELY(it == data_end_))
                it = data_begin_;

            value_at_it = first_try;
        }
    }

    [[nodiscard]] constexpr value_type build_bucket(node_type a, node_type b, unsigned tid = -1) const {
        es::swap_if(a > b, a, b);

        assert(a < (node_type(1) << kBitsPerNode));
        assert(b < (node_type(1) << kBitsPerNode));

        return build_bucket(static_cast<value_type>(a) | (static_cast<value_type>(b) << kBitsPerNode), tid);
    }

    [[nodiscard]] constexpr value_type build_bucket(value_type unlocked, unsigned tid = -1) const {
        return get_payload(unlocked) | (static_cast<value_type>(tid) << kBitsPerPayload);
    }

    [[nodiscard]] constexpr value_type get_payload(value_type tmp) const {
        return tmp & ((value_type(1) << kBitsPerPayload) - 1);
    }

    [[nodiscard]] constexpr unsigned get_lock(value_type tmp) const {
        return static_cast<unsigned>(tmp >> kBitsPerPayload);
    }

    [[nodiscard]] constexpr bool is_locked(value_type tmp) const {
        return get_lock(tmp) != get_lock(kEmpty);
    }

    [[nodiscard]] size_t hash(value_type value) const noexcept {
        if constexpr(sizeof(value) > 8) {
            static_assert(kBitsPerPayload <= 96);
            auto l = _mm_crc32_u64(hash_bias_, static_cast<uint64_t>(value));
            auto m = _mm_crc32_u64(l, static_cast<uint64_t>(value >> 16));
            auto h = _mm_crc32_u64(m, static_cast<uint64_t>(value >> 32));
            return static_cast<uint64_t>(value) ^ (l + 0x31411 * m + 0x87654321 * h);
        } else {
            auto l = _mm_crc32_u64(hash_bias_, static_cast<uint64_t>(value));
            auto h = _mm_crc32_u64(l, static_cast<uint64_t>(value >> 32));
            return value ^ (0x1234567 * l + 0x87654321 * h);
        }
    }


};
