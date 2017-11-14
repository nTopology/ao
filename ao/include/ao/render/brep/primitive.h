// ============================================================================
// Copyright 2017 nTopology Inc. All Rights Reserved.
// 
// Author: Yitzhak
// 
// ===========================================================================

#pragma once
#pragma warning(push, 0)
#include <Eigen/eigen>
#include <vector>
#pragma warning( pop )
#include "ao/render/brep/region.hpp"
#include "ao/eval/interval.hpp"

namespace Kernel {

//interface class for primitives that can be used to construct a tree.
//eval_deriv and eval_affine have not been made compatible with these yet, since they are not yet in the dependencies
//of ao.cpp.

class Primitive {
public:
  template<unsigned N>
  virtual Interval :: I getRange (Region<N> region) const = 0; 
    //If the range is larger than the actual range taken over the region,
    //everything will still work, but tighter ranges have better performance.

  virtual std::vector<Eigen::Vector3f> getGradient(Eigen::Vector3f point) const = 0;
      //Ordinarily, this will pass only Vector3d, but it can pass more if the primitive
      //has a non-continuous gradient at the given point (or the gradient is undefined with finitely many limit
      //values.)

  virtual double getValue(Eigen::Vector3f point) const = 0;

};

//These are implemented by the three coordinate axes (the default primitives):

class XPos : public Primitive {
public:
  template<N>
  Interval :: I getRange(Region<N> region) const override {
    return { region.lower.x(), region.upper.x() };
  }
  std::vector<Eigen::Vector3f> getGradient(Eigen::Vector3f point) const override {
    return std::vector<Eigen::Vector3f>{Eigen::Vector3f(1.f, 0.f, 0.f)};
  }
  double getValue(Eigen::Vector3f point) const override {
    return point.x();
  }
};

class YPos : public Primitive {
public:
  template<unsigned N>
  Interval :: I getRange(Region<N> region) const override {
    return { region.lower.y(), region.upper.y() };
  }
  std::vector<Eigen::Vector3f> getGradient(Eigen::Vector3f point) const override {
    return std::vector<Eigen::Vector3f>{Eigen::Vector3f(0.f, 1.f, 0.f)};
  }
  double getValue(Eigen::Vector3f point) const override {
    return point.y();
  }
};

class ZPos : public Primitive {
public:
  template<unsigned N>
  Interval :: I getRange(Region<N> region) const override {
    if (N == 3)
      return { region.lower.z(), region.upper.z() };
    else
      return { region.perp(0), region.perp(0) }; //A one-point interval.
  }
  std::vector<Eigen::Vector3f> getGradient(Eigen::Vector3f point) const override {
    return std::vector<Eigen::Vector3f>{Eigen::Vector3f(0.f, 0.f, 1.f)};
  }
  double getValue(Eigen::Vector3f point) const override {
    return point.z();
  }
};

} //Namespace Kernel