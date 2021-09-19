#pragma once

#include <assert.h>
#include <cstdint>
#include <random>

namespace linear_congruential_map {
    using node_t = std::uint32_t;

    static node_t inverse_of_in(node_t a, node_t p) {
        node_t t = 0;
        node_t new_t = 1;
        node_t r = p;
        node_t new_r = a;

        while (new_r != 0) {
            node_t tmp;
            const auto quot = static_cast<node_t>(r / new_r);

            tmp = t;
            t = new_t;
            new_t = tmp - quot * new_t;
            tmp = r;
            r = new_r;
            new_r = tmp - quot * new_r;
        }

        return t;
    }

    static bool is_prime(const node_t n) {
        if (n <= 1)
            return false;
        if (n <= 3)
            return true;
        if (n % 2 == 0 || n % 3 == 0)
            return false;

        node_t tmp = 5;
        while (tmp * tmp <= n) {
            if (n % tmp == 0 || n % (tmp + 2) == 0)
                return false;
            tmp = tmp + 6;
        }

        return true;
    }

    static node_t next_prime(const node_t n) {
        node_t next_prime = n + 1;

        if (next_prime <= 2)
            next_prime = 2;

        if (next_prime % 2 == 0)
            next_prime++;

        for (; !is_prime(next_prime); next_prime += 2);

        assert(next_prime > n);

        return next_prime;
    }

    class LinearCongruentialMap {
    private:
        node_t a;
        node_t a_inverse;
        node_t b;
        node_t p;

    public:
        LinearCongruentialMap(const node_t a_, const node_t b_, const node_t p_) :
                a(a_ % p_),
                a_inverse(inverse_of_in(a_, p_)),
                b(b_ % p_),
                p(p_) {
            assert(a_inverse > 0);
            assert(a > 0);
            assert(b < p);
        }

        node_t operator() (node_t node) const {
            return static_cast<node_t>((static_cast<uint64_t>(a) * node + b) % p);
        }

        [[nodiscard]] node_t invert(node_t hashed_node) const {
            return ((hashed_node - b) * a_inverse) % p;
        }

        template <typename Gen>
        static LinearCongruentialMap get_map(uint32_t p, Gen& gen) {
            std::uniform_int_distribution<node_t> dis(1, p - 1);
            return LinearCongruentialMap{dis(gen), dis(gen), p};
        }
    };
}
