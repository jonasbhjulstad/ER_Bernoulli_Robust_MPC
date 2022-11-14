#define EIGEN_USE_SYCL
#include <Eigen/Dense>
#include <sycl/sycl.hpp>
#include <FROLS_Execution.hpp>
static constexpr size_t N = 100;
int main()
{
    std::vector<Eigen::Matrix<float,N,N>> As;
    std::for_each(FROLS::execution::par_unseq, As.begin(), As.end(), [](auto& A) { A.setRandom(); });

}