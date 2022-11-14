#define COIN_DEBUG
#include <coin/ClpSimplex.hpp>
#include <FROLS_Math.hpp>
#include <random>


void solve()
{
    //generate a vector with random doubles between 20 and 40
    size_t N_rows = 5000;

    std::vector<double> v(N_rows);
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dis(20, 40);
    std::generate(v.begin(), v.end(), [&](){ return -dis(gen); });


    //s.t x + 2y + 3z <= 4

    CoinPackedMatrix matrix;
    ClpSimplex model;
    // matrix.setDimensions(N_rows, 2*N_rows + 2);
    matrix.setDimensions(0, 2*N_rows + 2);
    std::vector<char> sense = { 'H' };



    double tau = 0.05;

    std::vector<double> x(N_rows, 1);
    std::vector<double> y(N_rows, 1);
    y[0] = 1;
    for (int i = 1; i < N_rows; i++)
    {
        y[i] = v[i];
    }
    //read y from y.txt
    // std::ifstream y_file("y.txt");
    // for (int i = 0; i < N_rows; i++)
    // {
    //     y_file >> y[i];
    // }
    std::vector<double> objective(N_rows*2 + 2);
    for (int i = 0; i < N_rows; i++) {
      objective[i] = tau;
      objective[i + N_rows] = 1 - tau;
    }
    objective[N_rows*2] = 0;
    objective[N_rows*2 + 1] = 0;

    // CoinPackedMatrix matrix;
    // //reserve space for sparse N_rows x (2*N_rows + 2) matrix
    // matrix.setDimensions(N_rows, N_rows*2 + 2);

    // matrix.reserve(N_rows, N_rows*2);
    int theta_pos_idx = 2*N_rows;
    int theta_neg_idx = 2*N_rows + 1;
    const auto u_pos_idx = FROLS::range(0, N_rows);
    const auto u_neg_idx = FROLS::range(N_rows, 2*N_rows);
    std::vector<double> rhs(N_rows);
    std::vector<std::array<double, 4>> constraint_coeffs(N_rows);
    std::vector<std::array<int, 4>> constraint_indices(N_rows);

    std::vector<double> constraint_ub(N_rows);
    std::vector<double> constraint_lb(N_rows);
    for (int i = 0; i < N_rows; i++)
    {   
        constraint_ub[i] = y[i];
        constraint_lb[i] = y[i];
        constraint_coeffs[i] = {1, -1, x[i], -x[i]};
        constraint_indices[i] = {(int)u_pos_idx[i], (int)u_neg_idx[i],theta_pos_idx, theta_neg_idx};
        matrix.appendRow(4, &constraint_indices[i][0], &constraint_coeffs[i][0]);
    }

    model.setOptimizationDirection(1.);
    model.loadProblem(matrix, NULL, NULL, &objective[0], constraint_lb.data(), constraint_ub.data(), NULL);
    
    // model.setLogLevel(0);
    model.primal();
    std::cout << "Objective value: " << model.objectiveValue() << std::endl;
    std::cout << "theta: " << model.primalColumnSolution()[theta_pos_idx] - model.primalColumnSolution()[theta_neg_idx] << std::endl;
    //solve status
    int solve_status = model.status();
    std::cout << "Solve status: " << solve_status << std::endl;
    //check if solve was success
    if (model.status() != 0)
    {
        std::cout << "Error: " << model.status() << std::endl;
    }

}

int main()
{
   
   solve();

}