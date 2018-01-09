// ============================================================================
// Copyright 2017 nTopology Inc. All Rights Reserved. (provided we get ao under a different license)
// 
// Author: Yitzhak
// 
// ===========================================================================

#pragma once
#include <memory>
#include <array>

namespace Kernel {

struct PartialOctree {
  std::array<std::unique_ptr<PartialOctree>, 8> children;  //This is all there is unless inherited; the data consists of which children are nullptr.
  virtual ~PartialOctree() = default;  //Allows children to safely take derived classes.
};

} //namespace Kernel