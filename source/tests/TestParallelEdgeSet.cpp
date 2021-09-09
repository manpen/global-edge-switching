#include <es/ParallelEdgeSet.hpp>

#include <unordered_set>
#include <random>

#include <tlx/die.hpp>

void sequential_test() {
    std::mt19937_64 gen{0};

    for(size_t n : {10, 100, 1000}) {
        std::uniform_int_distribution<unsigned> uniform(0u, n-1);

        ParallelEdgeSet<> dut(n);
        std::unordered_set<unsigned> ref;

        for(auto m = 10 * n - 1; m; --m) {
            if (m % n == 0) {
                dut.rebuild();
            }

            auto val = uniform(gen);
            auto is_new = ref.insert(val).second;

            auto acq = dut.acquire(val, val, 0);

            if (is_new) {
                die_if(acq.first);
                die_if(acq.second != nullptr);
                auto res = dut.insert(val, val);
                die_if(res == nullptr);

            } else {
                die_unless(dut.insert(val, val) == nullptr);
                die_unless(acq.first);
                die_unless(acq.second != nullptr);

                dut.erase_and_release(acq.second);
                ref.erase(val);
            }
        }
    }
}


int main() {
    sequential_test();

    return 0;
}