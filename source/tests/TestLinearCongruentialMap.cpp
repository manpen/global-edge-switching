#include <es/LinearCongruentialMap.hpp>
#include <tlx/die.hpp>
#include <random>
#include <iostream>

void linear_congruential_map_test() {
    using hashmap = linear_congruential_map::LinearCongruentialMap;

    const uint32_t m = 1e6;
    const uint32_t m_prime = linear_congruential_map::next_prime(m);
    std::random_device rd;
    std::mt19937_64 gen(rd());

    for (int i = 0; i < 10; i++) {
        const hashmap h = linear_congruential_map::LinearCongruentialMap::get_map(m_prime, gen);
        std::vector<uint32_t> hashes;
        hashes.reserve(m);
        for (uint32_t j = 0; j < m; j++) {
            die_unless(h(j) < m_prime);
            hashes.emplace_back(h(j));
        }
        std::sort(hashes.begin(), hashes.end());
        die_unless(std::adjacent_find(hashes.begin(), hashes.end()) == hashes.end());
    }
}

int main() {
    linear_congruential_map_test();

    return 0;
}