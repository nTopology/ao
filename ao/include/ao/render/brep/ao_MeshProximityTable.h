// ============================================================================
// Copyright 2017 nTopology Inc. All Rights Reserved.
//
// Author: Yitzhak
//
// ===========================================================================

#pragma once
#include <glm/glm.hpp>
#pragma warning(push, 0)
#include <Eigen/Eigen>
#pragma warning( pop )

//interface classes for icarus_MeshProximityTable, to avoid dependency loops.

namespace Kernel {
class MeshProximityTableInterface {
public:
  class NodeInterface;
  class LeafNodeInterface;
  class BranchNodeInterface;
  virtual NodeInterface* getRootInterface() = 0;
};

class MeshProximityTableInterface::NodeInterface {
public:
  virtual glm::vec3 getMinimalPoint() const = 0;
  virtual glm::vec3 getMaximalPoint() const = 0;
  virtual float getMinimumSignedSquaredDistance() const = 0;
  virtual float getMaximumSignedSquaredDistance() const = 0;
  //Compares the distance from the first argument to the original manifold to the second argument.
  //-1 if the distance is less than the second argument, 0 if equal, 1 if more.
  virtual int compareDistance(const glm::vec3 point, const float distance)  const = 0;
  virtual float getSignedSquaredDistance(const glm::vec3 point) const = 0;
  virtual bool isLeaf() const = 0;
};

class MeshProximityTableInterface::LeafNodeInterface : public virtual MeshProximityTableInterface::NodeInterface {
public:
  virtual bool isInsideIfBoundary(glm::vec3 point, float offset, float max_err = 1e-8) const = 0;
  //if getSignedSquaredDistance is not between (max_err + offset)^2 and (max_err - offset)^2, return value is undefined.
  //Otherwise, adapts ao's isInside from eval_feature.
  virtual std::vector<Eigen::Vector4d> getNormalizedDerivsAndValues(const glm::vec3 point) const = 0;
};

class MeshProximityTableInterface::BranchNodeInterface : public virtual MeshProximityTableInterface::NodeInterface {
public:
  virtual const NodeInterface& getChild(int i) const = 0;
};
} //namespace Kernel
