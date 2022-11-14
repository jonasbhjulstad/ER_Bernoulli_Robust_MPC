#include "Regressor.hpp"
#include <FROLS_Eigen.hpp>
#include <FROLS_Execution.hpp>
#include <fmt/format.h>

namespace FROLS::Regression
{

  Regressor::Regressor(const Regressor_Param &p)
      : tol(p.tol), theta_tol(p.theta_tol), N_terms_max(p.N_terms_max), f_ERR_tol(p.f_ERR_tol) {}

<<<<<<< HEAD

    std::vector<Feature> Regressor::single_fit(const Mat &X, const Vec &y) const {
        uint32_t N_features = X.cols();
        Mat Q_global = Mat::Zero(X.rows(), N_features);
        Mat Q_current = Q_global;
        Mat A = Mat::Zero(N_features, N_features);
        Vec g = Vec::Zero(N_features);
        std::vector<Feature> best_features;
        best_features.reserve(N_terms_max);
        uint32_t end_idx = N_features;

        // fmt::print("Max features: {}\n", N_terms_max);
        // Perform one feature selection iteration for each feature
        for (int j = 0; j < N_features; j++) {
            // fmt::print("Feature {}\n", j+1);
            // Compute remaining variance by orthogonalizing the current feature
            Q_current =
                    used_feature_orthogonalize(X, Q_global, best_features);
            // Determine the best feature to add to the feature set
            Feature f = best_feature_select(Q_current, Q_global, y, best_features);

            if ((f.tag == FEATURE_INVALID) || (j >= (N_terms_max))) {
                end_idx = j;
                break;
            }
            best_features.push_back(f);
            Q_global.col(j) = Q_current.col(best_features[j].index);


            g[j] = best_features[j].g;

            for (int m = 0; m < j; m++) {
                A(m, j) = cov_normalize(Q_global.col(m), X.col(best_features[j].index));
            }
            A(j, j) = 1;

            // If ERR-tolerance is met, return non-orthogonalized parameters
            if (tolerance_check(Q_global.leftCols(j +1), y, best_features)) {
                end_idx = j+1;
                break;
            }


            Q_current.setZero();
        }
        theta_solve(A.topLeftCorner(end_idx, end_idx), g.head(end_idx), X, y, best_features);

        return best_features;
    }

    std::vector<Feature>
    Regressor::single_fit(const Mat &X, const Vec &y, std::vector<Feature> preselect_features) const {

        std::for_each(preselect_features.begin(), preselect_features.end(), [](auto& f){f.tag = FEATURE_PRESELECTED;});
        Vec y_diff = y;
        std::for_each(preselect_features.begin(), preselect_features.end(), [&](const auto& f){
            y_diff -= X.col(f.index)*f.theta;
        });
        Mat X_unused = X;

        std::vector<Feature> result;
        result.insert(result.begin(), preselect_features.begin(), preselect_features.end());
        std::vector<Feature> identified_features = single_fit(X_unused, y_diff);
        result.insert(result.end(), identified_features.begin(), identified_features.end());
        return result;

    }

    Feature Regressor::best_feature_select(const Mat &X, const Mat& Q_global, crVec &y, const std::vector<Feature> &used_features) const {
        const std::vector<Feature> candidates = candidate_regression(X, Q_global, y, used_features);
        std::vector<Feature> thresholded_candidates;
        std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(thresholded_candidates),
                     [&](const auto &f) {
                         return abs(f.g) > theta_tol;
                     });
        static bool warn_msg = true;

        Feature res;
        switch (thresholded_candidates.size()) {
            case 0:
                if (warn_msg)
                    std::cout << "[Regressor] Warning: threshold is too high for candidates" << std::endl;
                warn_msg = false;
                break;
            case 1:
                res = thresholded_candidates[0];
                break;
            default:
                res = feature_selection_criteria(thresholded_candidates);
                break;
        }

        return res;
    }

    std::vector<Feature> Regressor::fit(const Mat &X, crVec &y) {
        if ((X.rows() != y.rows())) {
            throw std::invalid_argument("X, U and y must have same number of rows");
        }
        std::vector<Feature> result;
        result = this->single_fit(X, y);
        return result;
    }

    Vec Regressor::predict(const Mat &Q, const std::vector<Feature> &features) const {
        Vec y_pred(Q.rows());
        y_pred.setZero();
        uint32_t i = 0;
        for (const auto &feature: features) {
            if (feature.f_ERR == -1) {
                break;
            }
            y_pred += Q.col(i) * feature.g;
            i++;
        }
        return y_pred;
    }

    struct Fit_Data {
        std::vector<Feature> preselect_features;
        Vec y;
    };


    std::vector<Feature> Regressor::transform_fit(const Mat &X_raw, const Mat &U_raw, crVec &y,
                                  Features::Feature_Model &model) {
        Mat XU(X_raw.rows(), X_raw.cols() + U_raw.cols());
        XU << X_raw, U_raw;
        Mat X = model.transform(XU);
        return single_fit(X, y);
    }

    std::vector<std::vector<Feature>> Regressor::transform_fit(const Regression_Data& rd,
                                  Features::Feature_Model &model) {
        
        Mat XU(rd.X.rows(), rd.X.cols() + rd.U.cols());
        XU << rd.X, rd.U;
        Mat X = model.transform(XU);
        std::vector<std::vector<Feature>> results(rd.Y.cols());
        std::transform(rd.Y.colwise().begin(), rd.Y.colwise().end(), results.begin(), [&](const Vec& y)
        {
            return single_fit(X, y);
        });
        return results;
    }


    std::vector<Feature> Regressor::transform_fit(const std::vector<std::string>& filenames, const std::vector<std::string>& colnames_x, const std::vector<std::string>& colnames_u, const std::string& colname_y, Features::Feature_Model& model)
=======
  std::vector<Feature>
  Regressor::single_fit(const std::vector<Mat> &X_list,
                        const std::vector<Vec> &y_list) const
  {
    uint32_t N_features = X_list[0].cols();
    uint32_t N_rows = X_list[0].rows();
    uint32_t N_timeseries = X_list.size();
    std::vector<Mat> Q_list_global(N_timeseries);
    std::vector<Mat> Q_list_current(N_timeseries);
    for (int i = 0; i < N_timeseries; i++)
>>>>>>> master
    {
      Q_list_global[i].resize(N_rows, N_features);
      Q_list_global[i].setZero();
      Q_list_current[i].resize(N_rows, N_features);
    }

    Mat A = Mat::Zero(N_features, N_features);
    Vec g = Vec::Zero(N_features);
    std::vector<Feature> best_features;
    best_features.reserve(N_terms_max);
    uint32_t end_idx = 0;

    // fmt::print("Max features: {}\n", N_terms_max);
    // Perform one feature selection iteration for each feature
    for (int j = 0; j < N_features; j++)
    {
      // fmt::print("Feature {}\n", j+1);
      // Compute remaining variance by orthogonalizing the current feature
      // std::transform(X_list.begin(), X_list.end(), Q_list_current.begin(),
      // [&Q_list_global](const Mat &X) { return used_feature_orthogonalize(X,
      // Q_global, best_features);});

      for (int i = 0; i < N_timeseries; i++)
      {
        Q_list_current[i] = used_feature_orthogonalize(
            X_list[i], Q_list_global[i], best_features);
      }
      // used_feature_orthogonalize(X, Q_global, best_features);
      // Determine the best feature to add to the feature set
      Feature f = best_feature_select(Q_list_current, Q_list_global, y_list,
                                      best_features);
      if (f.tag == FEATURE_INVALID)
      {
        end_idx = j + 1;
        break;
      }

      best_features.push_back(f);
      for (int i = 0; i < N_timeseries; i++)
      {
        Q_list_global[i].col(j) = Q_list_current[i].col(best_features[j].index);
      }
      // Q_global.col(j) = Q_current.col(best_features[j].index);

      g[j] = best_features[j].g;

      for (int m = 0; m < j; m++)
      {
        int a_mj_avg = 0;
        for (int i = 0; i < N_timeseries; i++)
        {
          a_mj_avg += cov_normalize(Q_list_global[i].col(m),
                                    X_list[i].col(best_features[j].index));
        }
        A(m, j) = a_mj_avg / N_timeseries;
      }
      A(j, j) = 1;

      // If ERR-tolerance is met, return non-orthogonalized parameters
      if (tolerance_check(Q_list_global, y_list, best_features, j + 1) ||
          (j == (N_terms_max - 1)))
      {
        end_idx = j + 1;
        break;
      }

      std::for_each(Q_list_current.begin(), Q_list_current.end(),
                    [](Mat &Q)
                    { Q.setZero(); });
    }
    theta_solve(A.topLeftCorner(end_idx, end_idx), g.head(end_idx),
                best_features);
    return best_features;
  }

  Feature Regressor::best_feature_select(
      const std::vector<Mat> &X_list, const std::vector<Mat> &Q_list_global,
      const std::vector<Vec> &y_list,
      const std::vector<Feature> &used_features) const
  {
    const std::vector<std::vector<Feature>> candidates =
        candidate_regression(X_list, Q_list_global, y_list, used_features);
    static bool warn_msg = true;

    Feature res;
    switch (candidates[0].size())
    {
    case 0:
      if (warn_msg)
        std::cout << "[Regressor] Warning: threshold is too high for candidates"
                  << std::endl;
      warn_msg = false;
      break;
    default:
      res = feature_selection_criteria(candidates);
      break;
    }

    return res;
  }

  Vec Regressor::predict(const Mat &Q,
                         const std::vector<Feature> &features) const
  {
    Vec y_pred(Q.rows());
    y_pred.setZero();
    uint32_t i = 0;
    for (const auto &feature : features)
    {
      if (feature.f_ERR == -1)
      {
        break;
      }
      y_pred += Q.col(i) * feature.g;
      i++;
    }
    return y_pred;
  }

  std::vector<std::vector<Feature>> Regressor::transform_fit(
      const std::vector<Mat> &X_raw, const std::vector<Mat> &U_raw,
      const std::vector<Mat> &Y_list, Features::Feature_Model &model)
  {
    uint32_t N_unfiltered_timeseries = X_raw.size();
    std::vector<Mat> X_filtered;
    std::vector<Mat> U_filtered;
    std::vector<Mat> Y_filtered;
    X_filtered.reserve(X_raw.size());
    U_filtered.reserve(U_raw.size());
    Y_filtered.reserve(Y_list.size());
    for (int i = 0; i < N_unfiltered_timeseries; i++)
    {
      bool y_valid = !std::any_of(Y_list[i].colwise().begin(), Y_list[i].colwise().end(),
                                  [](const Vec &y)
                                  { return y.isZero(); });
      if (y_valid)
      {
        X_filtered.push_back(X_raw[i]);
        U_filtered.push_back(U_raw[i]);
        Y_filtered.push_back(Y_list[i]);
      }
    }
    // copy Y_list into Y_list_filtered if y Vec is not zero

    uint32_t N_timeseries = X_filtered.size();
    uint32_t N_response = Y_filtered[0].cols();
    std::vector<Mat> XU_list(N_timeseries);
    for (int i = 0; i < N_timeseries; i++)
    {
      XU_list[i] = Mat::Zero(X_filtered[i].rows(), X_filtered[i].cols() + U_filtered[i].cols());
      XU_list[i] << X_filtered[i], U_filtered[i];
    }
    std::vector<Mat> X_list(N_timeseries);
    std::transform(XU_list.begin(), XU_list.end(), X_list.begin(),
                   [&model](const Mat &XU)
                   { return model.transform(XU); });

    std::vector<std::vector<Feature>> feature_list(N_response);
    for (int i = 0; i < N_response; i++)
    {
      feature_list[i].reserve(N_terms_max);
    }
    for (int j = 0; j < N_response; j++)
    {
      std::vector<Vec> y_list(N_timeseries);
      for (int i = 0; i < N_timeseries; i++)
      {
        y_list[i] = Y_filtered[i].col(j);
      }
      feature_list[j] = single_fit(X_list, y_list);
    }

    return feature_list;
  }

  std::vector<Feature> Regressor::transform_fit(
      const std::vector<Mat> &X_raw, const std::vector<Mat> &U_raw,
      const std::vector<Vec> &y_list, Features::Feature_Model &model)
  {
    uint32_t N_unfiltered_timeseries = X_raw.size();
    std::vector<Mat> X_filtered;
    std::vector<Mat> U_filtered;
    std::vector<Vec> y_filtered;
    X_filtered.reserve(X_raw.size());
    U_filtered.reserve(U_raw.size());
    y_filtered.reserve(y_list.size());
    for (int i = 0; i < N_unfiltered_timeseries; i++)
    {
      bool y_valid = !y_list[i].isZero();
      if (y_valid)
      {
        X_filtered.push_back(X_raw[i]);
        U_filtered.push_back(U_raw[i]);
        y_filtered.push_back(y_list[i]);
      }
    }

    uint32_t N_timeseries = X_filtered.size();
    std::vector<Mat> XU_list(N_timeseries);
    for (int i = 0; i < N_timeseries; i++)
    {
      XU_list[i] = Mat::Zero(X_filtered[i].rows(), X_filtered[i].cols() + U_filtered[i].cols());
      XU_list[i] << X_filtered[i], U_filtered[i];
    }
    std::vector<Mat> X_list(N_timeseries);
    std::transform(XU_list.begin(), XU_list.end(), X_list.begin(),
                   [&model](const Mat &XU)
                   { return model.transform(XU); });

    return single_fit(X_list, y_filtered);
  }

  std::vector<std::vector<Feature>>
  Regressor::transform_fit(const Regression_Data &rd,
                           Features::Feature_Model &model)
  {
    return transform_fit(rd.X, rd.U, rd.Y, model);
  }

  Feature Regressor::feature_selection_criteria(
      const std::vector<std::vector<Feature>> &features) const
  {
    uint32_t N_timeseries = features.size();
    uint32_t N_features = features[0].size();
    std::vector<std::pair<float, bool>> ERRs(N_features, {0, true});
    std::vector<uint32_t> N_ERRs(N_features,0);
    for (int i = 0; i < N_timeseries; i++)
    {   
      for (int j = 0; j < N_features; j++)
      {
        auto err = (features[i][j].tag == FEATURE_REGRESSION) && (features[i][j].f_ERR > 0) ? features[i][j].f_ERR : 0;
        ERRs[j].first += err;
        if (features[i][j].tag == FEATURE_INVALID)
        {
          ERRs[j].second = false;
        }
        else
        {
          N_ERRs[j]++;
        }
      }
    }

    std::for_each(ERRs.begin(), ERRs.end(),
                  [&, n = 0](std::pair<float, bool> &ERR)mutable
                  { ERR.first /= N_ERRs[n]; n++;});

    assert(!std::all_of(ERRs.begin(), ERRs.end(), [](const auto &p)
                        { return p.second == false; }));

    // find index of best ERR according to objective_condition with std::
    auto best_index = std::distance(ERRs.begin(), std::min_element(ERRs.begin(), ERRs.end(),
                                                                   [&](const auto &f0, const auto &f1)
                                                                   {
                                                                     return (f0.second) && objective_condition(f0.first, f1.first);
                                                                   }));
    //print all ERRs on one line
    for (int i = 0; i < N_features; i++)
    {
      std::cout << ERRs[i].first << " ";
    }
    std::cout << std::endl;
    Feature best_avg_feature{};
    best_avg_feature.f_ERR = 0;
    for (int i = 0; i < N_timeseries; i++)
    {
      if (features[i][best_index].tag == FEATURE_REGRESSION)
      {
        best_avg_feature.g += features[i][best_index].g;
      }
    }

    best_avg_feature.g /= N_timeseries;
    best_avg_feature.f_ERR = ERRs[best_index].first;
    best_avg_feature.index = features[0][best_index].index;
    best_avg_feature.tag = FEATURE_REGRESSION;
    if (best_avg_feature.f_ERR < f_ERR_tol)
    {
      best_avg_feature.tag = FEATURE_INVALID;
    }
    return best_avg_feature;
  }

  std::vector<uint32_t>
  Regressor::unused_feature_indices(const std::vector<Feature> &features,
                                    uint32_t N_features) const
  {
    std::vector<uint32_t> used_idx(features.size());
    std::transform(features.begin(), features.end(), used_idx.begin(),
                   [&](auto &f)
                   { return f.index; });
    return filtered_range(used_idx, 0, N_features);
  }

  int Regressor::regressor_count = 0;

} // namespace FROLS::Regression
