#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli.hpp>
#include <Sycl_Graph/Graph/Graph_Generation.hpp>
#include <Sycl_Graph/random.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <filesystem>
int main()
{

    using namespace Sycl_Graph::Sycl::Network_Models;
    using Sycl_Graph::Sycl::Network_Models::generate_erdos_renyi;
    using namespace Sycl_Graph::Network_Models;
    size_t N_pop = 100;
    float p_ER = 1;
    sycl::queue q;
    SIR_Graph G(q, 101, 100000);
    SIR_Bernoulli_Network sir(G, 0.1, 0.001);
    //generate sir_param
    size_t Nt = 100;
    std::vector<SIR_Bernoulli_Param<float>> sir_param(Nt);
    std::generate(sir_param.begin(), sir_param.end(), [&]() {
        return SIR_Bernoulli_Param<float>{0.05, 0.01, 100, 10};
    });
    std::cout << "Generating ER graph..." << std::endl;
    generate_erdos_renyi(G, N_pop, p_ER, SIR_INDIVIDUAL_S);
    std::cout << "Initializing..." << std::endl;
    sir.initialize();
    auto traj = sir.simulate(sir_param,Nt);
    //print traj
    for (auto &x : traj) {
        for (auto &y : x) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }

    //write to file
    std::ofstream file;
    std::filesystem::create_directory(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");
    file.open(std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/traj.csv");
    auto traj_T = Sycl_Graph::transpose(traj);

    for (auto &x : traj_T) {
        for (auto &y : x) {
            file << y << ",";
        }
        file << std::endl;
    }
    file.close();

}