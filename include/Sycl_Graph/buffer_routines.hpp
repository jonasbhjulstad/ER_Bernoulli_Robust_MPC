#ifndef SYCL_GRAPH_BUFFER_ROUTINES_HPP
#define SYCL_GRAPH_BUFFER_ROUTINES_HPP
#include <CL/sycl.hpp>
#include <vector>
namespace Sycl_Graph::Sycl
{
    template <typename T, typename uI_t = size_t>
    inline void host_buffer_copy(cl::sycl::buffer<T, 1>& buf, const std::vector<T>& vec, cl::sycl::queue& q, uI_t offset = 0)
    {
        //create buffer for vec
        cl::sycl::buffer<T, 1> vec_buf(vec.data(), vec.size());
        q.submit([&](cl::sycl::handler& h)
        {
        auto vec_acc = vec_buf.template get_access<sycl::access::mode::read>(h);
        auto acc = buf.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(vec.size(), [=](sycl::id<1> i)
        {
            acc[i + offset] = vec_acc[i];
        });
        });
    }
}
#endif