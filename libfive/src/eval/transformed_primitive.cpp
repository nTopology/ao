// ============================================================================
// Copyright 2017 nTopology Inc. All Rights Reserved. (provided we get ao under a different license)
// 
// Author: Yitzhak
// 
// ===========================================================================

#include "libfive/eval/transformed_primitive.h"

namespace Kernel {

Interval::I TransformedPrimitive::getRange(Region<2> region) const {
  auto xRange = xEvaluator.interval.eval(region.lower3().template cast<float>(), region.upper3().template cast<float>());
  auto yRange = yEvaluator.interval.eval(region.lower3().template cast<float>(), region.upper3().template cast<float>());
  auto zRange = zEvaluator.interval.eval(region.lower3().template cast<float>(), region.upper3().template cast<float>());
  return underlying.getRange(Region<3>({ xRange.lower(), yRange.lower(), zRange.lower() }, { xRange.upper(), yRange.upper(), zRange.upper() }));
}

Interval::I TransformedPrimitive::getRange(Region<3> region) const {
  auto xRange = xEvaluator.interval.eval(region.lower3().template cast<float>(), region.upper3().template cast<float>());
  auto yRange = yEvaluator.interval.eval(region.lower3().template cast<float>(), region.upper3().template cast<float>());
  auto zRange = zEvaluator.interval.eval(region.lower3().template cast<float>(), region.upper3().template cast<float>());
  return underlying.getRange(Region<3>({ xRange.lower(), yRange.lower(), zRange.lower() }, { xRange.upper(), yRange.upper(), zRange.upper() }));
}

std::vector<Eigen::Vector3f> TransformedPrimitive::getGradients(Eigen::Vector3f point) const {
  const auto xFeatures = xEvaluator.feature.featuresAt(point);
  const auto yFeatures = yEvaluator.feature.featuresAt(point);
  const auto zFeatures = zEvaluator.feature.featuresAt(point);
  std::vector<Eigen::Vector3f> out;
  std::vector<Eigen::Matrix3f> Jacobians;
  for (auto iterX = xFeatures.begin(); iterX != xFeatures.end(); ++iterX) {
    xEvaluator.feature.push(*iterX);
    for (auto iterY = xFeatures.begin(); iterY != xFeatures.end(); ++iterY) {
      yEvaluator.feature.push(*iterY);
      for (auto iterZ = xFeatures.begin(); iterZ != xFeatures.end(); ++iterZ) { 
        zEvaluator.feature.push(*iterZ);
        //This triple loop should almost never have substantial sizes for all 3; in the most common case,
        //all will be of size 1.
        if ((xFeatures.size() == 1 && yFeatures.size() == 1 && zFeatures.size() == 1) ||
          Feature::isCompatible<3>({ *iterX, *iterY, *iterZ })) {
          Eigen::Matrix3f Jacobian;
          Jacobian << xEvaluator.feature.deriv(point, *iterX), yEvaluator.feature.deriv(point, *iterY),
            zEvaluator.feature.deriv(point, *iterZ);
          Jacobians.push_back(Jacobian);
          zEvaluator.feature.pop();
        }
        yEvaluator.feature.pop();
      }
    }
    xEvaluator.feature.pop();
  }
  Eigen::Vector3f transformedPoint{ xEvaluator.feature.eval(point), yEvaluator.feature.eval(point), zEvaluator.feature.eval(point) };
  const auto underlyingGradients = underlying.getGradients(transformedPoint);
  out.reserve(Jacobians.size() * underlyingGradients.size());
  for (auto Jacobian : Jacobians) {
    for (auto gradient : underlyingGradients) {
      out.push_back(Jacobian * gradient);
    }
  }
  return out;
}

float TransformedPrimitive::getValue(Eigen::Vector3f point) const {
  Eigen::Vector3f transformedPoint{ xEvaluator.feature.eval(point), yEvaluator.feature.eval(point), zEvaluator.feature.eval(point) };
  return underlying.getValue(transformedPoint);
}

Primitive::prioritizationType TransformedPrimitive::getPrioritizationType(Eigen::Vector3f point) const {
  if (xEvaluator.feature.featuresAt(point).size() > 1 || yEvaluator.feature.featuresAt(point).size() > 1 || 
    zEvaluator.feature.featuresAt(point).size() > 1)
    return e_PRIMITIVE_NEITHER;
  else return underlying.getPrioritizationType(point);
}

}