//
// Created by arch on 9/14/22.
//

#ifndef SYCL_GRAPH_GRAPH_GENERATION_HPP
#define SYCL_GRAPH_GRAPH_GENERATION_HPP

#include <itertools.hpp>
#include <memory>
#include <CL/sycl.hpp>
#include <Sycl_Graph/random.hpp>
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Graph/Graph_Types_Sycl.hpp>
#include <iostream>

namespace Sycl_Graph::Sycl::Network_Models
{
    template <typename V, typename E, std::unsigned_integral uI_t, typename dType = float>
    void random_connect(Sycl_Graph::Sycl::Graph<V, E, uI_t> &G, dType p_ER, uI_t seed = 777)
    {
        Sycl_Graph::random::uniform_real_distribution d_ER;
        uI_t N_edges = 0;
        uI_t N_edges_max = Sycl_Graph::n_choose_k(G.N_vertices(), 2);
        //convert iter::combinations(Sycl_Graph::range(0, G.NV), 2) to a vector of dType-pairs

        std::vector<std::pair<uI_t, uI_t>> edge_idx;
        edge_idx.reserve(N_edges_max);
        //create rng
        Sycl_Graph::random::default_rng rng(seed);
        Sycl_Graph::random::uniform_real_distribution d(0, 1);
        for (auto &&v_idx : iter::combinations(Sycl_Graph::range(0, G.N_vertices()), 2))
        {
            bool connected = d(rng) < p_ER;
            uI_t to = (connected) ? v_idx[0] : G.invalid_id;
            uI_t from = (connected) ? v_idx[1] : G.invalid_id;
            edge_idx.push_back(std::pair<uI_t, uI_t>(to, from));
        }

        //create buffer of N_edges_max with seeds
        std::vector<uI_t> seeds(N_edges_max);
        std::generate(seeds.begin(), seeds.end(), std::rand);

        //load v_idx into buffer
        sycl::buffer<std::pair<uI_t, uI_t>> edge_idx_buf(edge_idx.data(), sycl::range<1>(edge_idx.size()));

        G.q.submit([&](sycl::handler &cgh) {
            auto edge_idx_acc = edge_idx_buf.template get_access<sycl::access::mode::read>(cgh);
            auto edges_acc = G.edge_buf.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for<class random_connect>(sycl::range<1>(N_edges_max), [=](sycl::id<1> idx) {
                edges_acc.to[idx] = edge_idx_acc[idx].first;
                edges_acc.from[idx] = edge_idx_acc[idx].second;
            });
        });
    }

    template <typename Graph, std::unsigned_integral uI_t, typename dType = float>
    void generate_erdos_renyi(
        Graph &G,
        uI_t N_pop,
        dType p_ER,
        const typename Graph::Vertex_Prop_t &node_prop)
    {
        auto vertex_ids = Sycl_Graph::range(0, N_pop);
        std::vector<typename Graph::Vertex_Prop_t> vertex_props(N_pop, node_prop);
        G.add(vertex_ids, vertex_props);
        random_connect(G, p_ER);
    }
}
#endif // FROLS_GRAPH_GENERATION_HPP
