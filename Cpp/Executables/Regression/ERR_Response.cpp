#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <ERR_Regressor.hpp>
#include <algorithm>
#include <FROLS_Path_Config.hpp>
#include <Regression_Algorithm.hpp>

std::string err_simulation_filename(size_t N_pop, double p_ER, size_t iter, std::string network_type) {
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/ERR_Simulation_" << network_type << "_" << N_pop << "_" << p_ER << "_" << iter
       << ".csv";
    return ss.str();
}

std::string SIR_Sine_filename(size_t N_pop, double p_ER, size_t iter, std::string network_type) {
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/SIR_Sine_Trajectory_Discrete_" << iter
       << ".csv";
    return ss.str();
}

std::string SIR_diff_filename(size_t N_pop, double p_ER, size_t iter, std::string network_type) {
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/Bernoulli_SIR_Delta_MC_" << N_pop << "_" << p_ER << "_" << iter
       << ".csv";
    return ss.str();
}



int main(int argc, char **argv) {
    const size_t Nx = 3;
    const std::string network_type = "SIR";
    const std::vector<std::string> colnames = {"S", "I", "R"};
    size_t N_sims = 7000; // 10000;
    size_t N_pop = 500;
    double p_ER = 1.0;
    using namespace FROLS;
    using namespace std::placeholders;
    size_t d_max = 1;
    size_t N_output_features = 16;
    size_t Nu = 1;
    auto Sine_fname_f = std::bind(SIR_Sine_filename, N_pop, p_ER, _1, network_type);
    auto diff_fname_f = std::bind(SIR_diff_filename, N_pop, p_ER, _1, network_type);

    auto MC_fname_f = std::bind(MC_filename, N_pop, p_ER, _1, network_type);
    auto outfile_f = std::bind(err_simulation_filename, N_pop, p_ER, _1, network_type);
    std::vector<std::vector<Feature>> preselected_features(4);
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);
    model.preselect("x0", 1.0, 0, FEATURE_PRESELECTED_IGNORE);
    model.preselect("x1", 1.0, 1, FEATURE_PRESELECTED_IGNORE);
    model.preselect("x2", 1.0, 2, FEATURE_PRESELECTED_IGNORE);
    std::vector<size_t> ignore_idx = {6, 8, 9, 13};
    std::for_each(ignore_idx.begin(), ignore_idx.end(), [&](auto& ig_idx){
        model.ignore(ig_idx);
    });
    FROLS::Regression::Regressor_Param reg_param;
    reg_param.tol = 1e-1;
    reg_param.theta_tol = 1e-10;
    reg_param.N_terms_max = 4;
    FROLS::Regression::ERR_Regressor regressor(reg_param);

    Regression::from_file_regression(MC_fname_f, {"S", "I", "R"}, {"p_I"}, N_sims, regressor, model, outfile_f);
    auto fnames = model.feature_names();
    std::for_each(fnames.begin(), fnames.end(), [n = 0](auto &name)mutable {
        std::cout << n << ": " << name << std::endl;
        n++;
    });


    return 0;
}
