#ifndef FROLS_Features_HPP
#define FROLS_Features_HPP

#include "Regressor.hpp"

namespace FROLS::Regression {
    struct ERR_Regressor : public Regressor {
        ERR_Regressor(const Regressor_Param& p) : Regressor(p){}

    private:
        std::vector<Feature> candidate_regression(const Mat &X,  const Mat& Q_global, const Vec &y,
                                            const std::vector<Feature> &used_features) const;

        bool tolerance_check(const Mat &Q, const Vec &y,
                             const std::vector<Feature> &best_features) const;

        Feature single_feature_regression(const Vec &x, const Vec &y) const;

        static bool best_feature_measure(const Feature&, const Feature&);
        Feature feature_selection_criteria(const std::vector<Feature> &features) const;

        void theta_solve(const Mat &A, const Vec &g, const Mat& X, const Vec& y, std::vector<Feature> &features) const;
    };
} // namespace FROLS::Regression

#endif