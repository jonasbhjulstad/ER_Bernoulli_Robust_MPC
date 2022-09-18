#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <Quantile_Regressor.hpp>
#include <ERR_Regressor.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

template<typename RNG>
Vec linsys_step(const Mat &A, const Mat &b, const Vec &x, const Vec &u, double b_std, RNG &rng) {
    std::normal_distribution<> d_b{0, b_std};
    Mat b_stochastic = b.array() + d_b(rng);
    return A * x + (b + b_stochastic) * u;
}
using namespace Eigen;
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");

int main() {
    using namespace FROLS;
    using namespace FROLS::Regression;
    using namespace FROLS::Features;
    size_t Nx = 3;
    Vec x0(Nx);
    x0 << 10, 100, -100;

    Mat A(Nx, Nx);
    A << .4, 0., 0, 0., .5, 0, 0., 0, .1;
    size_t Nu = 2;
    Mat b(Nx, Nu);
    b << 1,.2,0,.4,10,0;
    std::random_device rd{};
    std::mt19937 rng{rd()};
    double b_std = 1.;
    size_t Nt = 100;
    Mat u(Nt, Nu);
    size_t N_sims = 100;
    double omega = 10;
    std::vector<double> t(Nt+1);
    std::generate(t.begin(), t.end(), [&](){static size_t i = 0;
        return i++;});
    u.col(0) = Vec::LinSpaced(Nt, 0, M_PI_2 + .1);
    u.col(1) = Vec::LinSpaced(Nt, 0, M_PI);
    u = 10*u.array().sin();
//    u.setRandom();
//    u*=10;
    Mat Y(Nt * N_sims, Nx);
    Mat X(Nt * N_sims, Nx);
    Mat U(N_sims * Nt, Nu);
#pragma omp parallel for
    for (int j = 0; j < N_sims; j++) {
        Mat traj(Nt + 1, Nx);
        traj.row(0) = x0;
        std::ofstream f(FROLS_DATA_DIR + std::string("/Linear_Stochastic_Traj_") + std::to_string(j) + ".csv");
        for (int i = 0; i < Nt; i++) {
            traj.row(i + 1) = linsys_step(A, b, traj.row(i), u.row(i), b_std, rng);
        }
        f << traj.format(CSVFormat);
        f.close();
        Y(Eigen::seqN(j * Nt, Nt), Eigen::all) = traj.bottomRows(Nt);
        X(Eigen::seqN(j * Nt, Nt), Eigen::all) = traj.topRows(Nt);
        U(Eigen::seqN(j * Nt, Nt), Eigen::all) = u;
    }


    size_t d_max = 1;
    size_t N_features = 16;
    double ERR_tol = 0.2;
    double MAE_tol = 4;
    double tau = .95;
    double theta_tol = 1e-3;
    std::vector<size_t> ignore_idx = {0};

    Polynomial_Model model(Nx, Nu, N_features, d_max, ignore_idx);
    ERR_Regressor er(ERR_tol, theta_tol);
    Quantile_Regressor qr(tau, MAE_tol, theta_tol);


    er.transform_fit(X, U, Y, model);
    model.feature_summary();
    Mat X_sim_err = model.simulate(x0, u, Nt);
    qr.transform_fit(X, U, Y, model);
    model.feature_summary();
    Mat X_sim_qr = model.simulate(x0, u, Nt);;



    DataFrame df_qr, df_err;
    df_qr.assign({"S", "I", "R"}, X_sim_qr);
    std::vector<std::string> u_colnames = {"u0", "u1"};
    df_qr.assign(u_colnames, u);
    df_qr.assign("t", t);
    df_qr.write_csv(FROLS_DATA_DIR + std::string("/Linear_Stochastic_QR.csv"), ",");
    df_err.assign({"S", "I", "R"}, X_sim_err);
    df_err.assign(u_colnames, u);
    df_err.assign("t", t);

    df_err.write_csv(FROLS_DATA_DIR + std::string("/Linear_Stochastic_ERR.csv"), ",");





    return 0;
}