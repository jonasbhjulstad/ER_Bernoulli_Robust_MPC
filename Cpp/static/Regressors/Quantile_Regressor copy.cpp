#include "Quantile_Regressor.hpp"
#include <FROLS_Execution.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <limits>
namespace FROLS::Regression {
Quantile_Regressor::Quantile_Regressor(const Quantile_Param &p)
    : tau(p.tau), Regressor(p), solver_type(p.solver_type),
      problem_type(p.problem_type) {}

void Quantile_Regressor::theta_solve(const Mat &A, const Vec &g, const Mat &Q,
                                     const Vec &y,
                                     std::vector<Feature> &features) const {
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

Feature Quantile_Regressor::single_feature_regression(const Vec &x,
                                                      const Vec &y) const {
  {

    using namespace operations_research;
    uint32_t N_rows = x.rows();

    using namespace operations_research;
    MPSolver solver("Quantile_Solver", problem_type);
    const float infinity = solver.infinity();
    // theta_neg = solver.MakeNumVar(0.0, infinity, "theta_neg");
    // theta_pos = solver.MakeNumVar(0.0, infinity, "theta_pos");
    // operations_research::MPVariable *theta = solver.MakeNumVar(-infinity,
    // infinity, "theta");
    operations_research::MPVariable *theta_pos =
        solver.MakeNumVar(0, infinity, "theta_pos");
    operations_research::MPVariable *theta_neg =
        solver.MakeNumVar(0, infinity, "theta_neg");
    std::vector<operations_research::MPVariable *> u_pos;
    std::vector<operations_research::MPVariable *> u_neg;
    std::vector<operations_research::MPConstraint *> g;
    operations_research::MPObjective *objective;
    float sign = y.mean() > 0 ? 1.0 : -1.0;
    // float sign = 1.f;
    solver.MakeNumVarArray(N_rows, 0.0, infinity, "u_pos", &u_pos);
    solver.MakeNumVarArray(N_rows, 0.0, infinity, "u_neg", &u_neg);
    objective = solver.MutableObjective();
    objective->SetMinimization();
    float signed_tau = sign ? 1 - tau : tau;
    // std::for_each(u_pos.begin(), u_pos.end(),
    //             [=](auto u) { objective->SetCoefficient(u, signed_tau); });
    // std::for_each(u_neg.begin(), u_neg.end(),
    //             [=](auto u) { objective->SetCoefficient(u, 1 - signed_tau);
    //             });
    float W_u = 1000000.f;
    std::for_each(u_pos.begin(), u_pos.end(),
                  [=](auto u) { objective->SetCoefficient(u, W_u*tau); });
    std::for_each(u_neg.begin(), u_neg.end(),
                  [=](auto u) { objective->SetCoefficient(u, W_u*(1 - tau)); });
    g.resize(N_rows);
    std::for_each(g.begin(), g.end(),
                  [&](auto &gi) { gi = solver.MakeRowConstraint(); });

    for (int i = 0; i < N_rows; i++) {
      g[i]->SetCoefficient(theta_pos, x(i));
      g[i]->SetCoefficient(theta_neg, -x(i));
      g[i]->SetCoefficient(u_pos[i], 1);
      g[i]->SetCoefficient(u_neg[i], -1);
    //   g[i]->SetBounds(sign * y[i], sign * y[i]);
      g[i]->SetBounds(y[i], y[i]);
    }


    //            MX g = xi * (theta_pos - theta_neg) + u_pos - u_neg - dm_y;
    const bool solver_status = solver.Solve() == MPSolver::OPTIMAL;
    float f = objective->Value();
    // convert u_pos solution to Vec
    Vec u_pos_vec(N_rows);
    std::transform(u_pos.begin(), u_pos.end(), u_pos_vec.data(),
                   [](auto u) { return u->solution_value(); });
    Vec u_neg_vec(N_rows);
    std::transform(u_neg.begin(), u_neg.end(), u_neg_vec.data(),
                   [](auto u) { return u->solution_value(); });

    float theta_sol = theta_pos->solution_value() - theta_neg->solution_value();
    Feature candidate;
    candidate.f_ERR = std::numeric_limits<float>::infinity();
    solver.Clear();
    if (solver_status) {
      return Feature{f, theta_sol, 0, 0., FEATURE_REGRESSION};
    } else {
      std::cout << "[Quantile_Regressor] Warning: Quantile regression failed"
                << std::endl;
      return Feature{};
    }
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
  std::transform(candidate_idx.begin(),
                 candidate_idx.end(), candidates.begin(),
                 [=](const uint32_t &idx) {
                   Feature f = single_feature_regression(X.col(idx), y_diff);
                   f.index = idx;
                   return f;
                 });

  return candidates;
}

bool Quantile_Regressor::tolerance_check(
    const Mat &X, const Vec &y, const std::vector<Feature> &best_features) const {
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
