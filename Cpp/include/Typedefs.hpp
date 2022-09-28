#ifndef FROLS_EIGEN_TYPEDEFS_HPP
#define FROLS_EIGEN_TYPEDEFS_HPP
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <limits>
#include <map>
namespace FROLS {

typedef Eigen::Matrix<double, -1, -1> Mat;
typedef Eigen::Vector<double, -1> Vec;
typedef Eigen::Vector<int, -1> iVec;
typedef const Eigen::Ref<const Vec> crVec;
typedef const Eigen::Ref<const Mat> crMat;


enum Feature_Tag {FEATURE_INVALID = -1, FEATURE_REGRESSION, FEATURE_PRESELECTED, FEATURE_PRESELECTED_IGNORE};
const std::map<Feature_Tag, std::string> feature_tag_map = {{FEATURE_INVALID, "INVALID"}, {FEATURE_REGRESSION, "REGRESSION"}, {FEATURE_PRESELECTED, "PRESELECTED"}};
struct Feature {
  double f_ERR = -std::numeric_limits<double>::infinity(); // objective/Error Reduction Ratio
  double g;       // Feature (Orthogonalized Linear-in-the-parameters form)
  size_t index;   // Index of the feature in the original feature set
  double theta = 0;
  Feature_Tag tag = FEATURE_INVALID;
};
} // namespace FROLS
#endif