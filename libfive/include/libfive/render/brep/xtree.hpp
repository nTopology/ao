/*
libfive: a CAD kernel for modeling with implicit functions
Copyright (C) 2017  Matt Keeter

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
#pragma once

#include <array>
#include <atomic>
#include <iostream>
#include <mutex>

#include <cstdint>

#include <Eigen/Eigen>

#include "libfive/render/brep/region.hpp"
#include "libfive/render/brep/marching.hpp"
#include "libfive/render/brep/eval_xtree.hpp"
#include "libfive/eval/interval.hpp"
#include "libfive/render/axes.hpp"

namespace Kernel {

template <unsigned N>
class XTree
{
protected:
  //Nested structs hold data used to construct XTrees; they are to be created locally in the build function and passed to the constructor.

  struct ConstantBuildInfo;

  struct NodesToSplit {
    std::vector<XTree*> candidateTrees;
    std::mutex mMutex;
    void process(XTreeEvaluator* eval, ConstantBuildInfo& info);
  };

  struct ConstantBuildInfo { //Allows a large number of variables to be passed around by reference.
                             //"constant" refers to the fact that it doesn't change from node to node;
                             //some of its members are still mutable.
    const double min_feature;
    const double max_err;
    std::atomic_bool& cancel;
    NodesToSplit splittersHolder;
    ConstantBuildInfo(double min_feature, double max_err, std::atomic_bool& cancel) :
      min_feature(min_feature), max_err(max_err), cancel(cancel) { ; }
  };

public:
    /*
     *  Constructs an octree or quadtree by subdividing a region
     *  (unstoppable)
     */
    static std::unique_ptr<const XTree> build(
            Tree t, Region<N> region, double min_feature=0.1,
            double max_err=1e-8, bool multithread=true);

    /*
     *  Fully-specified XTree builder (stoppable through cancel)
     */
    static std::unique_ptr<const XTree> build(
            Tree t, const std::map<Tree::Id, float>& vars,
            Region<N> region, double min_feature,
            double max_err, bool multithread,
            std::atomic_bool& cancel);

    /*
     *  XTree builder that re-uses existing evaluators
     *  If multithread is true, es must be a pointer to an array of evaluators
     */
    static std::unique_ptr<const XTree> build(
            XTreeEvaluator* es,
            Region<N> region, double min_feature,
            double max_err, bool multithread,
            std::atomic_bool& cancel);

    /*
     *  Checks whether this tree splits
     */
    bool isBranch() const { return children[0].get() != nullptr; }

    /*
     *  Looks up a child, returning *this if this isn't a branch
     */
    const XTree<N>* child(unsigned i) const
    { return isBranch() ? children[i].get() : this; }

    /*
     *  Returns the filled / empty state for the ith corner
     */
    Interval::State cornerState(uint8_t i) const { return corners[i]; }

    /*
     *  Returns the corner position for the ith corner
     */
    Eigen::Array<double, N, 1> cornerPos(uint8_t i) const
    {
        Eigen::Array<double, N, 1> out;
        for (unsigned axis=0; axis < N; ++axis)
        {
            out(axis) = (i & (1 << axis)) ? region.upper(axis)
                                          : region.lower(axis);
        }
        return out;
    }

    /*
     *  Returns the averaged mass point
     */
    Eigen::Matrix<double, N, 1> massPoint() const;

    /*  Boilerplate for an object that contains an Eigen struct  */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /*
     *  Unpack the vertex into a 3-element array
     *  (using the region's perpendicular coordinates)
     */
    Eigen::Vector3d vert3(unsigned index=0) const;

    /*  The region filled by this XTree */
    const Region<N> region;

    /*  Children pointers, if this is a branch  */
    std::array<std::unique_ptr<XTree<N>>, 1 << N> children;

    /*  level = max(map(level, children)) + 1  
        depth = parent->depth + 1*/
    unsigned level=0;
    unsigned depth;

    /*  Vertex locations, if this is a leaf
     *
     *  To make cells manifold, we may store multiple vertices in a single
     *  leaf; see writeup in marching.cpp for details  */
    Eigen::Matrix<double, N, _pow(2, N - 1)> verts;

    /*
     *  Look up a particular vertex by index
     */
    Eigen::Matrix<double, N, 1> vert(unsigned i=0) const
    { assert(i < vertex_count); return verts.col(i); }

    /*  Array of filled states for the cell's corners
     *  (must only be FILLEd / EMPTY, not UNKNOWN or AMBIGUOUS ) */
    std::array<Interval::State, 1 << N> corners;

    /*  Leaf cell state, when known  */
    Interval::State type=Interval::UNKNOWN;

    /*  Feature rank for the cell's vertex, where                    *
     *      1 is face, 2 is edge, 3 is corner                        *
     *                                                               *
     *  This value is populated in find{Leaf|Branch}Matrices and     *
     *  used when merging intersections from lower-ranked children   */
    unsigned rank=0;

    /* Used as a unique per-vertex index when unpacking into a b-rep;   *
     * this is cheaper than storing a map of XTree* -> uint32_t         */
    mutable std::array<uint32_t, _pow(2, N - 1)> index;

    /*  Bitfield marking which corners are set */
    uint8_t corner_mask=0;

    /*  Stores the number of patches / vertices in this cell
     *  (which could be more than one to keep the surface manifold */
    unsigned vertex_count=0;

    /*  Marks whether this cell is combinable (manifold and does not have
    a descendant that needs to be split) or not  */
    bool combinable=false;

    /*  Single copy of the marching squares / cubes table, lazily
     *  initialized when needed */
    static std::unique_ptr<const Marching::MarchingTable<N>> mt;

protected:
    /*  Helper typedef for N-dimensional column vector */
    typedef Eigen::Matrix<double, N, 1> Vec;

    /*
     *  Private constructor for XTree
     *
     *  If multiple evaluators are provided, then tree construction will
     *  be distributed across multiple threads.
     */
    XTree(XTreeEvaluator* eval, Region<N> region,
          ConstantBuildInfo& info, bool multithread,
          XTree<N>* parent, uint8_t childNumberOfParent, 
          int depth);

    /*
     *  Searches for a vertex within the XTree cell, using the QEF matrices
     *  that are pre-populated in AtA, AtB, etc.
     *
     *  Minimizes the QEF towards mass_point
     *
     *  Stores the vertex in vert and returns the QEF error
     */
    double findVertex(unsigned i=0);

    /*
     *  Returns edges (as indices into corners)
     *  (must be specialized for a specific dimensionality)
     */
    const std::vector<std::pair<uint8_t, uint8_t>>& edges() const;

    /*
     *  Returns a table such that looking up a particular corner
     *  configuration returns whether that configuration is safe to
     *  collapse.
     *  (must be specialized for a specific dimensionality)
     *
     *  This implements the test from [Gerstner et al, 2000], as
     *  described in [Ju et al, 2002].
     */
    bool cornersAreManifold() const;

    /*
     *  Checks to make sure that the fine contour is topologically equivalent
     *  to the coarser contour by comparing signs in edges and faces
     *  (must be specialized for a specific dimensionality)
     *
     *  Returns true if the cell can be collapsed without changing topology
     *  (with respect to the leaves)
     */
    bool leafsAreManifold() const;

    //info is passed in case the neighbor does not exist 
    //and needs to be created by calling split.
    XTree<N>* neighbor(XTreeEvaluator* eval, ConstantBuildInfo& info, Axis::Axis A, bool D) const;

    /*  Mass point is the average intersection location *
     *  (the last coordinate is number of points summed) */
    Eigen::Matrix<double, N + 1, 1> _mass_point;

    /*  QEF matrices */
    Eigen::Matrix<double, N, N> AtA;
    Eigen::Matrix<double, N, 1> AtB;
    double BtB=0;
    XTree<N>* parent; //nullptr for root
    uint8_t childNumberOfParent; //if parent exists, parent->child(childNumberOfParent) returns *this.

    /*  Eigenvalue threshold for determining feature rank  */
    constexpr static double EIGENVALUE_CUTOFF=0.1f;

    //Forces the tree to split; 
    void split(XTreeEvaluator* eval, ConstantBuildInfo& info);

    //If it's not a branch yet, calls split to ensure there is a child; also returns a non-const pointer.
    XTree<N>& forceChild(XTreeEvaluator* eval, ConstantBuildInfo& info, unsigned i);

    //Should be called only on branches; tells whether this node's split was safe for the resulting mesh's topology,
    //or will require the neighboring face to split as well.
    bool faceSplitWasTopologicallySafe(Axis::Axis A, bool D) const;

};

// Explicit template instantiation declarations
template <> bool XTree<2>::cornersAreManifold() const;
template <> bool XTree<3>::cornersAreManifold() const;

template <> bool XTree<2>::leafsAreManifold() const;
template <> bool XTree<3>::leafsAreManifold() const;

template <> const std::vector<std::pair<uint8_t, uint8_t>>& XTree<2>::edges() const;
template <> const std::vector<std::pair<uint8_t, uint8_t>>& XTree<3>::edges() const;

template <> bool XTree<2>::faceSplitWasTopologicallySafe(Axis::Axis A, bool D) const;
template <> bool XTree<3>::faceSplitWasTopologicallySafe(Axis::Axis A, bool D) const;

extern template class XTree<2>;
extern template class XTree<3>;

}   // namespace Kernel
