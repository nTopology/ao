// ============================================================================
// Copyright 2017 nTopology Inc. All Rights Reserved. (provided we get ao under a different license)
// 
// Author: Yitzhak
// 
// ===========================================================================

#pragma once
#include "primitive.h"
#include "libfive/tree/tree.hpp"
#include "libfive/render/brep/eval_xtree.hpp"



namespace Kernel {

class TransformedPrimitive : public Primitive {
public:
  TransformedPrimitive(const Primitive& underlying, Tree X_, Tree Y_, Tree Z_) : 
    underlying(underlying), X_(X_), Y_(Y_), Z_(Z_), 
    xEvaluator(X_), yEvaluator(Y_), zEvaluator(Z_)
  { ; }
  Interval::I getRange(Region<2> region) const override;
  Interval::I getRange(Region<3> region) const override;
  std::vector<Eigen::Vector3f> getGradients(Eigen::Vector3f point) const override;
  float getValue(Eigen::Vector3f point) const override;
  prioritizationType getPrioritizationType(Eigen::Vector3f point) const override;
private:
  const Primitive& underlying;
  Tree X_;
  Tree Y_;
  Tree Z_;
  mutable XTreeEvaluator xEvaluator; //Used because it contains both an interval evaluator and a derivative evaluator.
  mutable XTreeEvaluator yEvaluator;
  mutable XTreeEvaluator zEvaluator;
};

}