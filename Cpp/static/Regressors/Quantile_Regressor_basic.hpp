#ifndef FROLS_QUANTILE_FEATURE_HPP
#define FROLS_QUANTILE_FEATURE_HPP

#include "Regressor.hpp"
#include <numeric>
#include <memory>
#include <thread>
#include "Simplex.hpp"


namespace FROLS::Regression
{
    struct Quantile_Param : public Regressor_Param
    {
        float tau = .95;
        uint32_t N_rows;
        uint32_t N_threads = 4;
        const std::string solver_type = "GLOP";
    };



    struct Quantile_Regressor : public Regressor
    {
        const float tau;

        Quantile_Regressor(const Quantile_Param &p);
        Feature feature_selection_criteria(const std::vector<Feature> &features) const;

        void theta_solve(const Mat &A, crVec &g, const Mat &X, crVec &y, std::vector<Feature> &features) const;

    private:
        Feature single_feature_regression(const Vec &x, const Vec &y) const;

        std::vector<Feature> candidate_regression(const Mat &X, const Mat &Q_global, crVec &y, const std::vector<Feature> &used_features) const;

        bool tolerance_check(const Mat &Q, crVec &y, const std::vector<Feature> &best_features) const;

        uint32_t feature_selection_idx = 0;
    };
}

#endif