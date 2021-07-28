#pragma once

#include <cassert>
#include <atomic>
#include <vector>

#include <nmmintrin.h>

#include <tlx/define/likely.hpp>
#include <tlx/math.hpp>

#include <tsl/robin_growth_policy.h>

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

    static constexpr size_t kBitsPerPayload = 2 * kBitsPerNode;

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

    // returns pair {found, ticket} --- the first entry is true iff the item is in the set;
    // the second entry returns a pointer if the bucket could be locked
    [[nodiscard]] std::pair<bool, iterator_type> acquire(node_type a, node_type b, unsigned tid) noexcept {
        auto reference_unlocked = build_bucket(a, b);
        auto reference_payload = get_payload(reference_unlocked);

        auto bucket = growth_.bucket_for_hash(hash(reference_payload));
        auto it = data_begin_ + bucket;

        while (true) {
            auto value = __atomic_load_n(it, __ATOMIC_CONSUME);

            if (value == kEmpty)
                return {false, nullptr};

            if (get_payload(value) == reference_payload) {
                // we found the correct bucket; let's try to lock it
                auto reference_locked = build_bucket(a, b, tid);
                auto did_acquire = __atomic_compare_exchange_n(it, &reference_unlocked, reference_locked, false, __ATOMIC_RELEASE,
                                                               __ATOMIC_CONSUME);

                return {true, did_acquire ? it : nullptr};
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

    // rebuilding
    void rebuild() {
        incpwl::ScopedTimer timer("rebuild");

        auto size = storage_.size() - 1;
        ParallelEdgeSet<Growth> new_instance(size, 1.0, hash_bias_ + 31);

        size_t empty_slots = 0;
        size_t num_items = 0;

        #pragma omp parallel for reduction(+:empty_slots, num_items)
        for (size_t i = 0; i != size; ++i) {
            auto value = data_begin_[i];
            empty_slots += (value == kEmpty);
            if (value == kEmpty || value == kDeleted)
                continue;

            assert(!is_locked(value));

            auto it = new_instance.insert(value);
            assert(it != nullptr);

            ++num_items;
        }

        *this = std::move(new_instance);

        if (0) {
            std::cout << "Load factor " << (100. * num_items / size) << "%. " << "Empty slots " << (100. * empty_slots / size) << "%\n";
        }
    }

    [[nodiscard]] size_t capacity() const noexcept {
        return max_size_;
    }

private:
    size_t max_size_;
    Growth growth_;

    uint64_t hash_bias_;

    std::vector<value_type> storage_;

    value_type *data_begin_;
    value_type *data_end_;

    [[nodiscard]] iterator_type insert(value_type reference_locked) noexcept {
        const auto reference_payload = get_payload(reference_locked);

        auto bucket = growth_.bucket_for_hash(hash(reference_payload));
        auto it = data_begin_ + bucket;

        while (true) {
            auto value = __atomic_load_n(it, __ATOMIC_CONSUME);

            if (value == kEmpty || value == kDeleted) {
                if (__atomic_compare_exchange_n(it, &value, reference_locked, true, __ATOMIC_RELEASE, __ATOMIC_CONSUME)) {
                    // we now have our foot in the door: we have a ticket, but still have to make sure
                    // that the value is not yet stored in the hash set -- so we search on; if we find
                    // it, we delete our temporary ticket, otherwise we make it permanent

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

                continue; // something went wrong; retry
            }

            if (get_payload(value) == reference_payload)
                return nullptr;

            ++it;
            if (TLX_UNLIKELY(it == data_end_))
                it = data_begin_;
        }
    }

    [[nodiscard]] constexpr value_type build_bucket(node_type a, node_type b, unsigned tid = -1) const {
        auto tmp = (a ^ b) * (a > b);
        a ^= tmp;
        b ^= tmp;

        assert(a < (node_type(1) << kBitsPerNode));
        assert(b < (node_type(1) << kBitsPerNode));


        return static_cast<value_type>(a) | (static_cast<value_type>(b) << kBitsPerNode) |
               (static_cast<value_type>(tid) << kBitsPerPayload);
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
            return static_cast<uint64_t>(value) ^ (l + 0x3141 * m + 0x87654321 * h);
        } else {
            auto l = _mm_crc32_u64(hash_bias_, static_cast<uint64_t>(value));
            auto h = _mm_crc32_u64(l, static_cast<uint64_t>(value >> 32));
            return value ^ (l + 0x87654321 * h);
        }
    }
};
