#ifndef FROLS_QUANTILE_FEATURE_HPP
#define FROLS_QUANTILE_FEATURE_HPP

#include "Regressor.hpp"
#include <memory>
#include <numeric>
#include <thread>
#include <ortools/linear_solver/linear_solver.h>

namespace FROLS::Regression {
struct Quantile_Param : public Regressor_Param {
  float tau = .95;
  uint32_t N_rows;
  uint32_t N_threads = 4;
  const float y_tol = 0.5;
  const float x_tol = std::numeric_limits<float>::epsilon();
  const std::string solver_type = "SCIP";
  operations_research::MPSolver::OptimizationProblemType problem_type =
      operations_research::MPSolver::OptimizationProblemType::
          GLOP_LINEAR_PROGRAMMING;
};

struct Quantile_Regressor : public Regressor {
  const float tau;
  const float y_tol;
  const float x_tol;
  Quantile_Regressor(const Quantile_Param &p);

<<<<<<< HEAD
  void theta_solve(const Mat &A, crVec &g, const Mat &X, crVec &y,
=======
  void theta_solve(const Mat &A, const Vec &g,
>>>>>>> master
                   std::vector<Feature> &features) const;

private:
  Feature single_feature_regression(const Vec &x, const Vec &y) const;

<<<<<<< HEAD
  std::vector<Feature>
  candidate_regression(const Mat &X, const Mat &Q_global, crVec &y,
                       const std::vector<Feature> &used_features) const;

  bool tolerance_check(const Mat &Q, crVec &y,
                       const std::vector<Feature> &best_features) const;
=======
  std::vector<std::vector<Feature>> candidate_regression(
      const std::vector<Mat> &X_list, const std::vector<Mat> &Q_list_global,
      const std::vector<Vec> &y_list,
      const std::vector<Feature> &used_features) const;
>>>>>>> master

  bool tolerance_check(
      const std::vector<Mat> &X_list, const std::vector<Vec> &y_list,
      const std::vector<Feature> &best_features, uint32_t) const;
  uint32_t feature_selection_idx = 0;

  operations_research::MPSolver::OptimizationProblemType problem_type;
  const std::string solver_type;
  bool objective_condition(float, float) const;

  // Quantile_LP construct_solver(uint32_t N_rows) const;
};
} // namespace FROLS::Regression

#endif