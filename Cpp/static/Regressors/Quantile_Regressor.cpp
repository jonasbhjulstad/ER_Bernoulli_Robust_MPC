#include "Quantile_Regressor.hpp"
#include <FROLS_Execution.hpp>
#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <limits>

namespace FROLS::Regression {
Quantile_Regressor::Quantile_Regressor(const Quantile_Param &p)
    : tau(p.tau), Regressor(p) {}

void Quantile_Regressor::theta_solve(const Mat &A, const Vec &g, const Mat &Q,
                                     const Vec &y,
                                     std::vector<Feature> &features) const {
if (features.size() > 0){
  std::vector<Feature> feature_tmp;
  feature_tmp.reserve(features.size());
  Mat y_diffs(Q.rows(), features.size() + 1);
  
  y_diffs.col(0) = y - Q.col(0) * features[0].g;
  features[0].theta = features[0].g;
    Vec coefficients = A.inverse() * g;
    for (int i = 0; i < coefficients.rows(); i++) {
      features[i].theta = coefficients[i];
    }
  }
}

Feature Quantile_Regressor::single_feature_regression(const Vec &x,
                                                      const Vec &y) const {

  uint32_t N_rows = x.rows();

  std::vector<double> objective(N_rows * 2 + 2);
  for (int i = 0; i < N_rows; i++) {
    objective[i] = tau;
    objective[i + N_rows] = 1 - tau;
  }
  objective[N_rows * 2] = 0;
  objective[N_rows * 2 + 1] = 0;

  // CoinPackedMatrix matrix;
  // //reserve space for sparse N_rows x (2*N_rows + 2) matrix
  // matrix.setDimensions(N_rows, N_rows*2 + 2);
  ClpSimplex model;
  // ClpInterior model;

  CoinPackedMatrix matrix;
  matrix.setDimensions(0, N_rows * 2 + 2);
  // matrix.reserve(N_rows, N_rows*2);
  // model.setLogLevel(1);
  int theta_pos_idx = 2 * N_rows;
  int theta_neg_idx = 2 * N_rows + 1;
  const auto u_pos_idx = FROLS::range(0, N_rows);
  const auto u_neg_idx = FROLS::range(N_rows, 2 * N_rows);
  std::vector<double> rhs(N_rows);
  std::vector<std::array<double, 4>> constraint_coeffs(N_rows);
  std::vector<std::array<int, 4>> constraint_indices(N_rows);

  std::vector<double> constraint_ub(N_rows);
  std::vector<double> constraint_lb(N_rows);
  for (int i = 0; i < N_rows; i++) {
    constraint_ub[i] = y[i];
    constraint_lb[i] = y[i];
    constraint_coeffs[i] = {1, -1, x[i], -x[i]};
    constraint_indices[i] = {(int)u_pos_idx[i], (int)u_neg_idx[i],
                             theta_pos_idx, theta_neg_idx};
    matrix.appendRow(4, &constraint_indices[i][0], &constraint_coeffs[i][0]);
  }

  model.setOptimizationDirection(1.);
  model.loadProblem(matrix, NULL, NULL, &objective[0], constraint_lb.data(),
                    constraint_ub.data(), NULL);

  model.primal();
  // model.primalDual();
  // model.setLogLevel(0);
  // solve status
  float f = model.objectiveValue();
  float g = model.primalColumnSolution()[theta_pos_idx] -
            model.primalColumnSolution()[theta_neg_idx];
  int solve_status = model.status();
  if (solve_status == 0) {
    return Feature{f, g, 0, 0.f, FEATURE_REGRESSION};
  } else {
    Feature res{};
    res.f_ERR = std::numeric_limits<float>::infinity();
    return res;
  }
}

std::vector<Feature> Quantile_Regressor::candidate_regression(
    const Mat &X, const Mat &Q_global, const Vec &y,
    const std::vector<Feature> &used_features) const {
  // get used indices of used_features
  std::vector<int> used_indices;

  used_indices.reserve(used_features.size());
  std::transform(used_features.begin(), used_features.end(),
                 std::back_inserter(used_indices),
                 [](const Feature &f) { return f.index; });

  Vec y_diff = y - predict(Q_global, used_features);

  static int counter = 0;

  std::vector<uint32_t> candidate_idx =
      unused_feature_indices(used_features, X.cols());
  std::vector<Feature> candidates(candidate_idx.size());
  std::transform(FROLS::execution::par_unseq, candidate_idx.begin(), candidate_idx.end(), candidates.begin(),
                 [=, this](const uint32_t &idx) {
                   Feature f = single_feature_regression(X.col(idx), y_diff);
                   f.index = idx;
                   return f;
                 });
  return candidates;
}

bool Quantile_Regressor::tolerance_check(
    const Mat &X, const Vec &y,
    const std::vector<Feature> &best_features) const {
  Vec y_pred = predict(X, best_features);
  Vec diff = y - y_pred;
  uint32_t N_samples = y.rows();
  float err = (diff.array() > 0).select(tau * diff, -(1 - tau) * diff).sum() /
              N_samples;
  bool no_improvement =
      (best_features.size() > 1) && (best_features.back().f_ERR < err);
  if (no_improvement) {
    // std::cout << "[Quantile_Regressor] Warning: Termination due to lack of
    // improvement" << std::endl;
    return true;
  } else if (err < tol) {
    // std::cout << "[Quantile_Regressor] Status: Successful tolerance
    // termination" << std::endl;
    return true;
  }
  // std::cout << "[Quantile_Regressor] Error: " << err << std::endl;
  return false;
}

Feature Quantile_Regressor::feature_selection_criteria(
    const std::vector<Feature> &features) const {
  return *std::min_element(
      features.begin(), features.end(),
      [](const Feature &f1, const Feature &f2) { return f1.f_ERR < f2.f_ERR; });
}

} // namespace FROLS::Regression
