#include "Regressor.hpp"
#include <omp.h>
namespace FROLS::Regression {

Regressor::Regressor(double tol, double theta_tol)
    : tol(tol), theta_tol(theta_tol), regressor_id(regressor_count),
      regressor_logger(spdlog::basic_logger_mt(
              ("regressor_logger_" + std::to_string(regressor_id)).c_str(),
          (std::string(FROLS_LOG_DIR) + "/regressor_" + std::to_string(regressor_id) + "_log.txt").c_str(), true)) {
  regressor_logger->set_level(spdlog::level::debug);
  regressor_logger->info("{:^15}{:^15}{:^15}{:^15}", "Feature", "g", "theta",
                       "f_ERR");
  regressor_count++;
}
Mat Regressor::used_feature_orthogonalize(const Mat &X, const Mat &Q,
                                          const std::vector<Feature>& used_features) const {
  size_t N_features = X.cols();
  size_t N_samples = X.rows();
  Mat Q_current = Mat::Zero(X.rows(), X.cols());
  for (int k = 0; k < N_features; k++) {
    if (std::none_of(used_features.begin(), used_features.end(), [&](const auto& feature){return feature.index == k;})) {
      Q_current.col(k) = vec_orthogonalize(X.col(k), Q.leftCols(used_features.size()));
    }
  }
  return Q_current;
}

std::vector<Feature> Regressor::single_fit(crMat X, crVec y) {
  size_t N_features = X.cols();
  Mat Q_global = Mat::Zero(X.rows(), N_features);
  Mat Q_current = Q_global;
  Mat A = Mat::Zero(N_features, N_features);
  Vec g = Vec::Zero(N_features);

  std::vector<Feature> best_features;
  size_t end_idx = N_features;
  // Perform one feature selection iteration for each feature
  for (int j = 0; j < N_features; j++) {
    // Compute remaining variance by orthogonalizing the current feature
    Q_current =
        used_feature_orthogonalize(X, Q_global, best_features);
    // Determine the best feature to add to the feature set
    best_features.push_back(feature_select(Q_current, y, best_features));
    g[j] = best_features[j].g;

    Q_global.col(j) = Q_current.col(best_features[j].index);
    for (int m = 0; m < j; m++) {
        A(m, j) = cov_normalize(Q_global.col(m), X.col(best_features[j].index));
    }
    A(j, j) = 1;

    // If ERR-tolerance is met, return non-orthogonalized parameters
    if (tolerance_check(Q_global.leftCols(j + 1), y, best_features)) {
      end_idx = j + 1;
      best_features.resize(end_idx);
      break;
    }

    Q_current.setZero();
  }
  Vec coefficients =
      A.topLeftCorner(end_idx, end_idx).inverse() * g.head(end_idx);
  // assign coefficients to features
  for (int i = 0; i < end_idx; i++) {
    best_features[i].theta = coefficients[i];
    regressor_logger->info("{:^15}{:^15.3f}{:^15.3f}{:^15.3f}", i,
                         best_features[i].g, best_features[i].theta,
                         best_features[i].f_ERR);
  }

  return best_features;
}

std::vector<std::vector<Feature>> Regressor::fit(crMat &X, crMat &Y) {
  if ((X.rows() != Y.rows())) {
    throw std::invalid_argument("X, U and Y must have same number of rows");
  }
  size_t N_response = Y.cols();
  std::vector<std::vector<Feature>> result(N_response);
  {
    for (int i = 0; i < N_response; i++) {
      result[i] = single_fit(X, Y.col(i));
    }
  }
  return result;
}
Vec Regressor::predict(crMat &Q, const std::vector<Feature>& features) const
{
  Vec y_pred(Q.rows());
  y_pred.setZero();
  size_t i = 0;
  for (const auto& feature: features)
  {
    if(feature.f_ERR == -1) {
        break;
    }
    y_pred += Q.col(i)*feature.g;
    i++;
  }
  return y_pred;
}

void Regressor::transform_fit(crMat &X_raw, crMat &U_raw, crMat &Y,
                              Features::Feature_Model &model){
  Mat XU(X_raw.rows(), X_raw.cols() + U_raw.cols());
  XU << X_raw, U_raw;
  Mat X = model.transform(XU);
  model.features = fit(X, Y);
}

int Regressor::regressor_count = 0;

} // namespace FROLS::Regression
