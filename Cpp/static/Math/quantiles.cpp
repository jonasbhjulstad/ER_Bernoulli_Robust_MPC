#include "quantiles.hpp"
#include <FROLS_Path_Config.hpp>
#include <FROLS_Math.hpp>
#include <algorithm>
#include <execution>
namespace FROLS {
    double quantile(std::vector<double> list, double tau) {
        typename std::vector<double>::iterator b = list.begin();
        typename std::vector<double>::iterator e = list.end();
        typename std::vector<double>::iterator quant = list.begin();
        const size_t pos = tau * std::distance(b, e);
        std::advance(quant, pos);

        std::nth_element(b, quant, e);
        return *quant;
    }

    std::vector<double> dataframe_quantiles(DataFrameStack &dfs,
                                            std::string col_name, double tau) {
        size_t N_rows = 0;
        for (int i = 0; i < dfs.get_N_frames(); i++) {
            N_rows = std::max({N_rows, (size_t) dfs[i].get_N_rows()});
        }
        size_t N_frames = dfs.get_N_frames();
        std::vector<double> result(N_rows);
        std::vector<double> xk;
        xk.reserve(N_rows);
        for (int i = 0; i < N_rows; i++) {
            for (int j = 0; j < N_frames; j++) {
                if (dfs[j].get_N_rows() > i) {
                    xk.push_back((*dfs[j][col_name])[i]);

                }
            }
            result[i] = quantile(xk, tau);
            xk.clear();
        }
        return result;
    }

    void quantiles_to_file(size_t N_simulations, const std::vector<std::string>& colnames, std::function<std::string(size_t)> MC_fname_f, std::function<std::string(size_t)> q_fname_f) {
        static std::thread::id thread_0 = std::this_thread::get_id();
        std::vector<std::string> filenames(N_simulations);
        size_t iter = 0;
        for (int i = 0; i < N_simulations; i++)
        {
            filenames[i] = MC_fname_f(i);
        }
        {
            using namespace FROLS;
            DataFrameStack dfs(filenames);
            size_t N_rows = dfs[0].get_N_rows();
            std::vector<double> t = (*dfs[0]["t"]);
            std::vector<double> xk(N_simulations);

            std::vector<double> quantiles = FROLS::arange(0.05, 0.95, 0.05);

            std::vector<std::vector<size_t>> q_trajectories(quantiles.size());
            for (auto &traj: q_trajectories) {
                traj.resize(N_rows);
            }
            for (int i = 0; i < q_trajectories.size(); i++) {
                if (std::this_thread::get_id() == thread_0)
                {
                    std::cout << "Quantile " << i+1 << " of " << q_trajectories.size() << std::endl;
                }
                DataFrame df;
                df.assign("t", t);

                std::for_each(std::execution::par_unseq, colnames.begin(), colnames.end(), [&](const auto& colname){df.assign(colname, dataframe_quantiles(dfs, colname));});
                df.write_csv(q_fname_f(i), ",");
            }
        }
    }
}