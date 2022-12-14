#include "Sycl_Graph/Math/math.hpp"

namespace Sycl_Graph {

    uint32_t n_choose_k(uint32_t n, uint32_t k) {
        if (k > n) return 0;
        if (k * 2 > n) k = n - k;
        if (k == 0) return 1;

        int result = n;
        for (int i = 2; i <= k; ++i) {
            result *= (n - i + 1);
            result /= i;
        }
        return result;
    }

    std::vector<float> linspace(float min, float max, int N) {
        std::vector<float> res(N);
        for (int i = 0; i < N; i++) {
            res[i] = min + (max - min) * i / (N - 1);
        }
        return res;
    }



    std::vector<uint32_t> range(uint32_t start, uint32_t end) {
        std::vector<uint32_t> res(end - start);
        for (uint32_t i = start; i < end; i++) {
            res[i - start] = i;
        }
        return res;
    }

    std::vector<uint32_t> filtered_range(const std::vector<uint32_t> &filter_idx, uint32_t min, uint32_t max) {
        auto full_range = range(min, max);
        std::vector<uint32_t> res;
        std::copy_if(full_range.begin(), full_range.end(), std::back_inserter(res),
                     [&](const uint32_t idx) {
                         return std::none_of(filter_idx.begin(), filter_idx.end(),
                                             [&](const uint32_t &f_idx) { return idx == f_idx; });
                     });
        return res;
    }



} // namespace Sycl_Graph
