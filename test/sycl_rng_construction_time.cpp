#include <CL/sycl.hpp>
#include <oneapi/dpl/random>
#include <iostream>

int main()
{
    sycl::property_list props{ sycl::property::queue::enable_profiling() };    
    // create gpu queue
    sycl::queue q(props);
    // create random number generator
    auto event = q.submit([&](sycl::handler &cgh) { // profile rng
        // create buffer
        sycl::buffer<uint32_t> rng_buf(1);
        // get access to buffer
        auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(cgh);
        // submit kernel
        cgh.single_task([=]()
                                     {
            //create rng
            oneapi::dpl::minstd_rand rng;
            //seed rng
            rng.seed(0);
            //generate random number
            rng_acc[0] = rng(); });

    });

    auto src = aligned_alloc(8, 1024);
    memset(src, 1, 1024);


    auto dst = sycl::malloc_device(1024,  q);
    auto ev2 = q.memset(dst, 0, 1024);

    auto ev3 = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(ev2);
        cgh.memcpy(dst, src, 1024);
    });

    event.wait();




    // time profile
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();

    std::cout << "Time taken to construct rng: " << (end - start) / 1000000.0 << " ms" << std::endl;

auto start3 = ev3.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end3 = ev3.get_profiling_info<sycl::info::event_profiling::command_end>();

    std::cout << "Time taken to construct rng: " << (end3 - start3) / 1000000.0 << " ms" << std::endl;
    return 0;




}