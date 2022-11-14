#ifndef FROLS_QUANTILE_FEATURE_HPP
#define FROLS_QUANTILE_FEATURE_HPP

#include "Regressor.hpp"
// #define COIN_DEBUG

#include <ClpSimplex.hpp>
#include <ClpInterior.hpp>

#include <memory>
#include <numeric>
#include <thread>

namespace FROLS::Regression {
struct Quantile_Param : public Regressor_Param {
  float tau = .95;
  uint32_t N_rows;
  uint32_t N_threads = 4;
};



struct Quantile_Regressor : public Regressor {
  const float tau;

  Quantile_Regressor(const Quantile_Param &p);
  Feature
  feature_selection_criteria(const std::vector<Feature> &features) const;

  void theta_solve(const Mat &A, const Vec &g, const Mat &X, const Vec &y,
                   std::vector<Feature> &features) const;

private:
  Feature single_feature_regression(const Vec &x, const Vec &y) const;

  std::vector<Feature>
  candidate_regression(const Mat &X, const Mat &Q_global, const Vec &y,
                       const std::vector<Feature> &used_features) const;

  bool tolerance_check(const Mat &Q, const Vec &y,
                       const std::vector<Feature> &best_features) const;

  uint32_t feature_selection_idx = 0;

  // Quantile_LP construct_solver(uint32_t N_rows) const;
};
} // namespace FROLS::Regression

#endif