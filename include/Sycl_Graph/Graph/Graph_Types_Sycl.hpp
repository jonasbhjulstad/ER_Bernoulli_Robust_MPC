// transform structs to have perfect forwarding

#ifndef Sycl_Graph_Graph_Types_Sycl_hpp
#define Sycl_Graph_Graph_Types_Sycl_hpp
#include <CL/sycl.hpp>
#include <Sycl_Graph/buffer_routines.hpp>
namespace Sycl_Graph::Sycl {
template <typename E, typename uI_t, cl::sycl::access::mode Mode>
struct Edge_Accessor {
  Edge_Accessor(cl::sycl::buffer<E, 1> &edge_buf,
                cl::sycl::buffer<uI_t, 1> &to_buf,
                cl::sycl::buffer<uI_t, 1> &from_buf, sycl::handler &h,
                cl::sycl::property_list props = {})
      : data(edge_buf, h, props), to(to_buf, h, props),
        from(from_buf, h, props) {}
  sycl::accessor<E, 1, Mode> data;
  sycl::accessor<uI_t, 1, Mode> to;
  sycl::accessor<uI_t, 1, Mode> from;
};

template <typename E, std::unsigned_integral uI_t> struct Edge_Buffer {
  // current number of edges
  uI_t N_edges = 0;
  // maximum number of edges
  const uI_t NE;
  cl::sycl::queue &q;
  cl::sycl::buffer<uI_t, 1> to_buf;
  cl::sycl::buffer<uI_t, 1> from_buf;
  cl::sycl::buffer<E, 1> data_buf;
  static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
  Edge_Buffer(uI_t NE, cl::sycl::queue &q,
              const cl::sycl::property_list &props = {})
      : to_buf(sycl::range<1>(NE), props), from_buf(sycl::range<1>(NE), props),
        data_buf(sycl::range<1>(NE), props), NE(NE), q(q) {}

  Edge_Buffer(std::vector<Edge<E, uI_t>> edges, cl::sycl::queue &q,
              const cl::sycl::property_list &props = {})
      : to_buf(sycl::range<1>(edges.size()), props),
        from_buf(sycl::range<1>(edges.size()), props),
        data_buf(sycl::range<1>(edges.size()), props), NE(edges.size()), q(q) {
        //create a buffer for edges
        sycl::buffer<Edge<E, uI_t>, 1> new_edge_buf(edges, props);


        // sycl::buffer<Edge<E, uI_t>, 1> new_edge_buf(edges.data(), sycl::range<1>(edges.size()), q, props);
        q.submit([&](sycl::handler &h){
          auto v_acc = this->get_access<sycl::access::mode::write>(h);
          auto new_v_acc = new_edge_buf.template get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>(edges.size()), [=](sycl::id<1> id) {
          v_acc.data[id] = new_v_acc[id].data;
          v_acc.to[id] = new_v_acc[id].to;
          v_acc.from[id] = new_v_acc[id].from;
        });});
  }
  template <cl::sycl::access::mode Mode>
  Edge_Accessor<E, uI_t, Mode> get_access(cl::sycl::handler &h) {
    return Edge_Accessor<E, uI_t, Mode>(data_buf, to_buf, from_buf, h);
  }

  void add(const std::vector<Edge<E, uI_t>> &edges) {

    Edge_Buffer<E, uI_t> new_edges(edges, q);
    add(new_edges);
  }

  void add(Edge_Buffer<E, uI_t>& new_edges)
  {
    // add edges to the end of the buffer
    uI_t offset = N_edges;
    q.submit([&](cl::sycl::handler &h) {
        auto e_acc = this->get_access<cl::sycl::access::mode::write>(h);
        auto new_e_acc = new_edges.get_access<cl::sycl::access::mode::read>(h);
      h.parallel_for<class edge_add>(cl::sycl::range<1>(new_edges.size()),
                                     [=](cl::sycl::id<1> id) {
                                        e_acc.to[id[0] + offset] = new_e_acc.to[id[0]];
                                        e_acc.from[id[0] + offset] = new_e_acc.from[id[0]];
                                        e_acc.data[id[0] + offset] = new_e_acc.data[id[0]];
                                     });
    });
    N_edges += new_edges.size();
  }

  void add(const std::vector<uI_t> &to, const std::vector<uI_t> &from,
           const std::vector<E> &data) {
    // add edges to the end of the buffer
    uI_t offset = N_edges;
    q.submit([&](cl::sycl::handler &h) {
      auto to_acc =
          to_buf.template get_access<cl::sycl::access::mode::read_write>(h);
      auto from_acc =
          from_buf.template get_access<cl::sycl::access::mode::read_write>(h);
      auto data_acc =
          data_buf.template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<class edge_add>(cl::sycl::range<1>(to.size()),
                                     [=](cl::sycl::id<1> id) {
                                       to_acc[id[0] + N_edges] = to[id[0]];
                                       from_acc[id[0] + N_edges] = from[id[0]];
                                       data_acc[id[0] + N_edges] = data[id[0]];
                                     });
    });
    N_edges += to.size();
  }

  void remove(const std::vector<uI_t> &to, const std::vector<uI_t> &from) {
    // Set ids of edges to invalid_id
    q.submit([&](cl::sycl::handler &h) {
      auto data_acc =
          data_buf.template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<class edge_remove>(
          cl::sycl::range<1>(to.size()), [=](cl::sycl::id<1> id) {
            auto it = std::find_if(data_acc.begin(), data_acc.end(),
                                   [id, from, to](const auto &e) {
                                     return e.from == from[id[0]] &&
                                            e.to == to[id[0]];
                                   });
            if (it != data_acc.end()) {
              it->from = invalid_id;
              it->to = invalid_id;
            }
          });
    });
    N_edges -= to.size();
  }

  uI_t size() const
  {
    return data_buf.size();
  }
};
template <typename V, typename uI_t, cl::sycl::access::mode mode>
struct Vertex_Accessor {
  Vertex_Accessor(cl::sycl::buffer<V, 1> &v_buf,
                  cl::sycl::buffer<uI_t, 1> &id_buf, sycl::handler &h,
                  cl::sycl::property_list props = {})
      : data(v_buf, h, props), id(id_buf, h, props) {}
  sycl::accessor<V, 1, mode> data;
  sycl::accessor<uI_t, 1, mode> id;
};

template <typename V, std::unsigned_integral uI_t> struct Vertex_Buffer {
  static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();

  uI_t N_vertices = 0;
  const uI_t NV;
  cl::sycl::buffer<uI_t, 1> id_buf;
  cl::sycl::buffer<V, 1> data_buf;
  cl::sycl::queue &q;
  Vertex_Buffer(uI_t NV, cl::sycl::queue &q,
                const cl::sycl::property_list &props = {})
      : id_buf(sycl::range<1>(NV), props), data_buf(sycl::range<1>(NV), props),
        NV(NV), q(q) {}

  Vertex_Buffer(const std::vector<Vertex<V, uI_t>> &vertices,
                cl::sycl::queue &q, const cl::sycl::property_list &props = {})
      : id_buf(sycl::range<1>(vertices.size()), props),
        data_buf(sycl::range<1>(vertices.size()), props), NV(vertices.size()),
        q(q) {
    std::vector<uI_t> ids(vertices.size());
    std::vector<V> v_data(vertices.size());
    for (uI_t i = 0; i < vertices.size(); ++i) {
      ids[i] = vertices[i].id;
      v_data[i] = vertices[i].data;
    }
    host_buffer_copy(id_buf, ids, q);
    host_buffer_copy(data_buf, v_data, q);
  }

  template <cl::sycl::access::mode mode>
  Vertex_Accessor<V, uI_t, mode> get_access(cl::sycl::handler &h) {
    return Vertex_Accessor<V, uI_t, mode>(data_buf, id_buf, h);
  }

  void add(const std::vector<Vertex<V, uI_t>> &vertices) {
    std::vector<uI_t> ids(vertices.size());
    std::vector<V> v_data(vertices.size());
    for (uI_t i = 0; i < vertices.size(); ++i) {
      ids[i] = vertices[i].id;
      v_data[i] = vertices[i].data;
    }
    host_buffer_copy(id_buf, ids, q);
    host_buffer_copy(data_buf, v_data, q);
    N_vertices += vertices.size();
  }

  void add(const std::vector<uI_t> &ids, const std::vector<V> &v_data) {
    Sycl_Graph::Sycl::host_buffer_copy(id_buf, ids, q, N_vertices);
    Sycl_Graph::Sycl::host_buffer_copy(data_buf, v_data, q, N_vertices);

    N_vertices += ids.size();
  }

  std::vector<V> get_data(const std::vector<uI_t> &ids) {
    std::vector<V> result(ids.size());

    cl::sycl::buffer<V, 1> res_buf(result.data(), ids.size());
    q.submit([&](cl::sycl::handler &h) {
      auto out = res_buf.template get_access<cl::sycl::access::mode::read>(h);
      auto v_acc =
          data_buf.template get_access<cl::sycl::access::mode::write>(h);
      h.parallel_for<class vertex_prop_search>(
          cl::sycl::range<1>(ids.size()), [=](cl::sycl::id<1> id) {
            auto it =
                std::find_if(v_acc.begin(), v_acc.end(),
                             [id](const auto &v) { return v.id == id[0]; });
            if (it != v_acc.end()) {
              out[id] = it->data;
            }
          });
    });
    return result;
  }

  void assign(const std::vector<uI_t> &id, const std::vector<V> &data) {
    q.submit([&](cl::sycl::handler &h) {
      auto v_acc =
          data_buf.template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<class vertex_assign>(
          cl::sycl::range<1>(id.size()), [=](cl::sycl::id<1> id) {
            auto it =
                std::find_if(v_acc.begin(), v_acc.end(),
                             [id](const auto &v) { return v.id == id[0]; });
            if (it != v_acc.end()) {
              it->data = data[id[0]];
            }
          });
    });
  }

  void remove(const std::vector<uI_t> &id) {
    q.submit([&](cl::sycl::handler &h) {
      auto v_acc =
          data_buf.template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<class vertex_remove>(
          cl::sycl::range<1>(id.size()), [=](cl::sycl::id<1> id) {
            auto it =
                std::find_if(v_acc.begin(), v_acc.end(),
                             [id](const auto &v) { return v.id == id[0]; });
            if (it != v_acc.end()) {
              it->id = invalid_id;
            }
          });
    });
    N_vertices -= id.size();
  }
};

} // namespace Sycl_Graph::Sycl
#endif