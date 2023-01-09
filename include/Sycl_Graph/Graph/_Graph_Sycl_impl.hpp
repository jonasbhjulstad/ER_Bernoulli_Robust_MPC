//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_GRAPH_SYCL_HPP
#define SYCL_GRAPH_GRAPH_SYCL_HPP
#ifdef SYCL_GRAPH_USE_SYCL
#include <algorithm>
#include <numeric>
#include <array>
#include <cassert>
#include <limits>
#include <mutex>
#include <iostream>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <iterator>
// #include <Sycl_Graph/execution.hpp>
#include <type_traits>
#include <CL/sycl.hpp>
#include "Graph_Types.hpp"
#include "Graph_Types_Sycl.hpp"
#include <Sycl_Graph/Algorithms/Algorithms.hpp>
#include <Sycl_Graph/data_containers.hpp>

namespace Sycl_Graph::Sycl
{
  
  template <typename V, typename E, std::unsigned_integral uI_t>
  struct Graph
  {
    // create copy constructor
    Graph(Graph &other) = default;
    Graph(cl::sycl::queue &q, uI_t NV, uI_t NE, const cl::sycl::property_list &props = {})
        : q(q), vertex_buf(NV, q, props), edge_buf(NE, q, props)
    {
    }

    Graph(cl::sycl::queue &q, const std::vector<Vertex<V, uI_t>>& vertices, const std::vector<Edge<E, uI_t>>& edges, const cl::sycl::property_list &props = {})
        : Graph(q), vertex_buf(vertices, q, props), edge_buf(edges, q, props)
    {
    }
    cl::sycl::queue &q;
    Vertex_Buffer<V, uI_t> vertex_buf;
    Edge_Buffer<E, uI_t> edge_buf;
    const uI_t& NV = vertex_buf.NV;
    const uI_t& NE = edge_buf.NE;
    using Vertex_t = Vertex<V, uI_t>;
    using Edge_t = Edge<E, uI_t>;
    using Vertex_Prop_t = V;
    using Edge_Prop_t = E;
    using uInt_t = uI_t;
    static constexpr auto invalid_id = std::numeric_limits<uI_t>::max();
    inline uI_t N_vertices() const {return vertex_buf.N_vertices;}
    inline uI_t N_edges() const {return edge_buf.N_edges;}

    uI_t find(auto condition)
    {
      uI_t idx = Vertex_t::invalid_id;
      cl::sycl::buffer<uI_t, 1> res_buf(&idx, 1);
      q.submit([&](cl::sycl::handler &h)
               {
        auto out = res_buf.template get_access<cl::sycl::access::mode::write>(h);
        auto vertex_acc = vertex_buf.template get_access<cl::sycl::access::mode::read>(h);
        find(out, vertex_acc, condition, h); });
    }

    uI_t find(auto &res_acc, auto &v_acc, auto condition, sycl::handler &h)
    {
      h.parallel_for<class vertex_id_search>(cl::sycl::range<1>(v_acc.size()), [=](cl::sycl::id<1> id)
                                             { if (condition(v_acc[id[0]])) res_acc[0] = id[0]; });
    }



    template <cl::sycl::access::mode mode>
    auto get_vertex_access(sycl::handler &h)
    {
      return vertex_buf.template get_access<mode>(h);
    }

    template <cl::sycl::access::mode mode>
    auto get_edge_access(sycl::handler &h)
    {
      return edge_buf.template get_access<mode>(h);
    }

    void add(const std::vector<Vertex<V, uI_t>> &vertices)
    {
      vertex_buf.add(vertices);
    }

    void add(const std::vector<uI_t> &ids, const std::vector<V> &v_data)
    {
      vertex_buf.add(ids, v_data);
    }

    void remove(const std::vector<uI_t> &id)
    {
      vertex_buf.remove(id);
    }
    void remove(const std::vector<uI_t> &to, const std::vector<uI_t> &from)
    {
      edge_buf.remove(to, from);
    }

    void add(const std::vector<uI_t> &to, const std::vector<uI_t> &from, const std::vector<E> &e_data)
    {
      edge_buf.add(to, from, e_data);
    }
    void add(const std::vector<Edge<E, uI_t>> &edges)
    {
      edge_buf.add(edges);
    }


    template <typename... Args>
    void assign_vertex(Args &&... args)
    {
      vertex_buf.assign(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void assign_edge(Args &&... args)
    {
      edge_buf.assign(std::forward<Args>(args)...);
    }

    template <typename... Args>
    V get_vertex(Args &&... args)
    {
      return vertex_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename... Args>
    E get_edge(Args &&... args)
    {
      return edge_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<V> get_vertex_data(Args &&... args)
    {
      return vertex_buf.get_data(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<E> get_edge_data(Args &&... args)
    {
      return edge_buf.get_data(std::forward<Args>(args)...);
    }

  };
}

#endif
#endif // Sycl_Graph_hpp
