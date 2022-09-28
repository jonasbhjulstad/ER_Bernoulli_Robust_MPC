#include "Feature_Model.hpp"
#include "Regressor.hpp"

namespace FROLS::Features {

    Feature_Model::Feature_Model(const size_t N_output_features) : N_output_features(
            N_output_features) {}

    Vec Feature_Model::transform(crMat &X_raw, size_t target_index, bool &index_failure) {
        return _transform(X_raw, target_index, index_failure);
    }

    Mat Feature_Model::transform(crMat &X_raw, const std::vector<Feature> preselected_features) {
        size_t N_input_features = X_raw.cols();
        size_t N_rows = X_raw.rows();
        Mat X_poly(N_rows, N_output_features + preselected_features.size());
        size_t feature_idx = 0;
        size_t col_iter = 0;
        bool index_failure = 0;
        candidate_feature_idx.reserve(N_output_features);
        candidate_feature_idx.clear();
        preselect_feature_idx.reserve(preselected_features.size());
        preselect_feature_idx.clear();
        do {
            bool ignored = std::any_of(ignore_idx.begin(), ignore_idx.end(),
                                       [&](const auto &ig_idx) { return feature_idx == ig_idx; })
                           ||
                           std::any_of(preselected_features.begin(), preselected_features.end(), [&](const Feature &f) {
                               return (feature_idx == f.index) && (f.tag == FEATURE_PRESELECTED_IGNORE);
                           });
            bool preselected = std::any_of(preselected_features.begin(), preselected_features.end(),
                                           [&](const auto &ps_feature) { return feature_idx == ps_feature.index; });
            if (!ignored) {

                X_poly.col(col_iter) = transform(X_raw, feature_idx, index_failure);
                if (!index_failure) {
                    col_iter++;
                    candidate_feature_idx.push_back(feature_idx);
                }
                if (preselected) {
                    preselect_feature_idx.push_back(feature_idx);
                }
            }
            feature_idx++;
        } while ((col_iter < (N_output_features + preselected_features.size())) && !index_failure);

        X_poly.conservativeResize(N_rows, col_iter);
        return X_poly;
    }


    Vec Feature_Model::step(crVec &x, crVec &u) {
        Vec x_next(x.rows());
        Mat X(1, x.rows() + u.rows());
        X << x.transpose(), u.transpose();
        bool index_failure = false;
        x_next.setZero();
        for (int i = 0; i < features.size(); i++) {
            for (int j = 0; j < features[i].size(); j++) {
                x_next(i) +=
                        features[i][j].theta *
                        _transform(X, candidate_feature_idx[features[i][j].index], index_failure).value();
                if (index_failure) {
                    break;
                }
            }
        }

        return x_next;
    }

    Mat Feature_Model::simulate(crVec &x0, crMat &U, size_t Nt) {
        Mat X(Nt + 1, x0.rows());
        X.setZero();
        X.row(0) = x0;
        for (int i = 0; i < Nt; i++) {
            X.row(i + 1) = step(X.row(i), U.row(i));
        }
        return X;
    }

    void Feature_Model::ignore(size_t feature_index) {
        ignore_idx.push_back(feature_index);
    }

    void Feature_Model::ignore(const std::string &feature_name) {
        ignore(get_feature_index(feature_name));
    }
    void Feature_Model::preselect(size_t feature_index, double theta, size_t response_index, Feature_Tag tag)
    {
        if (response_index >= preselected_features.size())
        {
            preselected_features.resize(response_index+1);
        }
        Feature f;
        f.tag = tag;
        f.theta = theta;
        f.index = feature_index;
        preselected_features[response_index].push_back(f);
    }
    double f_ERR = -std::numeric_limits<double>::infinity(); // objective/Error Reduction Ratio
    double g;       // Feature (Orthogonalized Linear-in-the-parameters form)
    size_t index;   // Index of the feature in the original feature set
    double theta = 0;
    Feature_Tag tag = FEATURE_INVALID;

    void Feature_Model::preselect(const std::string& feature_name, double theta, size_t response_index, Feature_Tag tag)
    {
        preselect(get_feature_index(feature_name), theta, response_index, tag);
    }


} // namespace FROLS::Features