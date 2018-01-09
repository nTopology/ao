// ============================================================================
// Copyright 2017 nTopology Inc. All Rights Reserved. (provided we get ao under a different license)
// 
// Author: Yitzhak
// 
// ===========================================================================

#pragma once
#pragma warning(push, 0)
#include <Eigen/eigen>
#include <set>
#pragma warning( pop )
#include "ao/render/brep/region.hpp"
#include "ao/eval/interval.hpp"

namespace Kernel {

//interface class for primitives that can be used to construct a tree.
//eval_deriv and eval_affine have not been made compatible with these yet, since they are not yet in the dependencies
//of ao.cpp.

class Primitive {
public:
  enum prioritizationType {e_PRIMITIVE_MINIMUM, e_PRIMITIVE_MAXIMUM, e_PRIMITIVE_NEITHER};

  virtual Interval :: I getRange (Region<2> region) const = 0; 
  virtual Interval :: I getRange (Region<3> region) const = 0;
    //If the range is larger than the actual range taken over the region,
    //everything will still work, but tighter ranges have better performance.

  virtual std::vector<Eigen::Vector3f> getGradients(Eigen::Vector3f point) const = 0;
      //Ordinarily, this will pass only Vector3d, but it can pass more if the primitive
      //has a non-continuous gradient at the given point (or the gradient is undefined with finitely many limit
      //values.)

  virtual float getValue(Eigen::Vector3f point) const = 0;

  virtual prioritizationType getPrioritizationType(Eigen::Vector3f point) const = 0;
      //Indicates whether to optimize features based on the assumption that it follows the branch that minimizes
      //value in that neighborhood, the one that maximizes the value, or not to optimize on such an assumption

  virtual ~Primitive() = default;
};

//These are implemented by the three coordinate axes (the default primitives):

class XPos : public Primitive {
public:
  Interval :: I getRange(Region<2> region) const override {
    return { region.lower.x(), region.upper.x() };
  }
  Interval::I getRange(Region<3> region) const override {
    return { region.lower.x(), region.upper.x() };
  }
  std::vector<Eigen::Vector3f> getGradients(Eigen::Vector3f point) const override {
    return std::vector<Eigen::Vector3f>{Eigen::Vector3f(1.f, 0.f, 0.f)};
  }
  float getValue(Eigen::Vector3f point) const override {
    return point.x();
  }
  prioritizationType getPrioritizationType(Eigen::Vector3f point) const override {
    return e_PRIMITIVE_NEITHER;
  }
};

class YPos : public Primitive {
public:
  Interval :: I getRange(Region<2> region) const override {
    return { region.lower.y(), region.upper.y() };
  }
  Interval::I getRange(Region<3> region) const override {
    return { region.lower.y(), region.upper.y() };
  }
  std::vector<Eigen::Vector3f> getGradients(Eigen::Vector3f point) const override {
    return std::vector<Eigen::Vector3f>{Eigen::Vector3f(0.f, 1.f, 0.f)};
  }
  float getValue(Eigen::Vector3f point) const override {
    return point.y();
  }
  prioritizationType getPrioritizationType(Eigen::Vector3f point) const override {
    return e_PRIMITIVE_NEITHER;
  }
};

class ZPos : public Primitive {
public:
  Interval :: I getRange(Region<2> region) const override {
      return { region.perp(0), region.perp(0) }; //A one-point interval.
  }
  Interval::I getRange(Region<3> region) const override {
      return { region.lower.z(), region.upper.z() };
  }
  std::vector<Eigen::Vector3f> getGradients(Eigen::Vector3f point) const override {
    return std::vector<Eigen::Vector3f>{Eigen::Vector3f(0.f, 0.f, 1.f)};
  }
  float getValue(Eigen::Vector3f point) const override {
    return point.z();
  }
  prioritizationType getPrioritizationType(Eigen::Vector3f point) const override {
    return e_PRIMITIVE_NEITHER;
  }
};

} //Namespace Kernel