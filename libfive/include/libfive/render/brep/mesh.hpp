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

#include "libfive/tree/tree.hpp"

#include "libfive/render/axes.hpp"
#include "libfive/render/brep/region.hpp"
#include "libfive/render/brep/brep.hpp"
#include "libfive/render/brep/xtree.hpp"
#include <boost/container/small_vector.hpp>

namespace Kernel {

class Mesh : public BRep<3> {
public:
    /*
     *  Blocking, unstoppable render function
     */
    static std::unique_ptr<Mesh> render(const Tree t, const Region<3>& r,
                                        double min_feature=0.1, double max_err=1e-8);

    /*
     *  Fully-specified render function
     */
    static std::unique_ptr<Mesh> render(
            const Tree t, const std::map<Tree::Id, float>& vars,
            const Region<3>& r, double min_feature, double max_err,
            std::atomic_bool& cancel);

    /*
     *  Render function that re-uses evaluators
     *  es must be a pointer to at least eight Evaluators
     */
    static std::unique_ptr<Mesh> render(
            XTreeEvaluator* es,
            const Region<3>& r, double min_feature, double max_err,
            std::atomic_bool& cancel);

    /*
     *  Writes the mesh to a file
     */
    bool saveSTL(const std::string& filename, bool isBinary = true);

    /*
     *  Merge multiple bodies and write them to a single file
     */
    static bool saveSTL(const std::string& filename,
                        const std::list<const Mesh*>& meshes,
                        bool isBinary);

    /*
     *  Called by Dual::walk to construct the triangle mesh
     */
    template <Axis::Axis A, bool D>
    void load(const std::array<const XTree<3>*, 4>& ts);

protected:
    /*  Walks an XTree, returning a mesh  */
    static std::unique_ptr<Mesh> mesh(std::unique_ptr<const XTree<3>> tree,
                                      std::atomic_bool& cancel);

    /*
     *  Inserts a line into the mesh as a zero-size triangle
     *  (used for debugging)
     */
    void line(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

    void addTriangle(Eigen::Matrix<uint32_t, 3, 1> vertices, Axis::Axis A, bool D, bool force = false);
    //The third vertex is generally the one that is "too close" to the original edge.  
    //Force forces it to be a positive triangle.

    void removeTriangle(uint32_t target);

    void processNegativeTriangles();

private:
    struct negativeTriangle {
        Eigen::Matrix<uint32_t, 3, 1> vertices;
        //In the same orientation as if it were added as a normal triangle, but the vertex that is misplaced 
        //compared to the line between the other two (extended by the axis) is always last.
        Axis::Axis A;
        bool D;
    };

    std::vector<negativeTriangle> negativeTriangles;
    //std::unordered_map<std::array<uint32_t, 2>, std::array<int32_t, 2>, boost::hash<std::array<uint32_t, 2>>> edgesToBranes;
    //Removed because it was getting stuck somewhere in the hash function.

    std::map<std::array<uint32_t, 2>, boost::container::small_vector<size_t, 1>> edgesToBranes;
    //For faster processing of negative triangles.  The doubled value is because (dual) marching cubes can sometimes produce non-manifold shapes
    //that have 4 triangles along a single edge, 2 in each direction.  If only one value is used, it should be value[0], and value[1] should be marked -1.
    std::vector<std::pair<Axis::Axis, bool>> positiveTriangleExtraInfo; //Is needed to process negative triangles, but can be cleared afterward.

    void testEdgesToBranes() {//for debugging
        auto idx = 0;
        for (auto iter = branes.begin(); iter != branes.end(); ++iter, ++idx) {
            for (auto i = 0; i < 3; ++i) {
                if (edgesToBranes.find({ (*iter)(i), (*iter)((i + 1) % 3) }) == edgesToBranes.end())
                    abort();
                auto values = edgesToBranes.find({ (*iter)(i), (*iter)((i + 1) % 3) })->second;
                auto found = false;
                for (auto iter2 = values.begin(); iter2 != values.end(); ++iter) {
                    if (*iter2 == idx) {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    abort();
            }
        }
    }

    void testEdgeMatching() {
        std::map<std::array<uint32_t, 2>, int> counts;
        for (auto iter = branes.begin(); iter != branes.end(); ++iter) {
            ++counts[{(*iter)(0), (*iter)(1)}];
            ++counts[{(*iter)(1), (*iter)(2)}];
            ++counts[{(*iter)(2), (*iter)(0)}];
            --counts[{(*iter)(1), (*iter)(0)}];
            --counts[{(*iter)(2), (*iter)(1)}];
            --counts[{(*iter)(0), (*iter)(2)}];
        }
        for (auto iter = negativeTriangles.begin(); iter != negativeTriangles.end(); ++iter) {
            ++counts[{(iter->vertices)(0), (iter->vertices)(1)}];
            ++counts[{(iter->vertices)(1), (iter->vertices)(2)}];
            ++counts[{(iter->vertices)(2), (iter->vertices)(0)}];
            --counts[{(iter->vertices)(1), (iter->vertices)(0)}];
            --counts[{(iter->vertices)(2), (iter->vertices)(1)}];
            --counts[{(iter->vertices)(0), (iter->vertices)(2)}];
        }
        for (auto iter = counts.begin(); iter != counts.end(); ++iter) {
            if (iter->second != 0)
                abort();
        }
    }
};

}   // namespace Kernel
