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
#include <numeric>
#include <fstream>
#include <boost/algorithm/string/predicate.hpp>

#include "libfive/render/brep/mesh.hpp"
#include "libfive/render/brep/xtree.hpp"
#include "libfive/render/brep/dual.hpp"

namespace Kernel {

template <Axis::Axis A, bool D>
void Mesh::load(const std::array<const XTree<3>*, 4>& ts)
{
    int es[4];
    {   // Unpack edge vertex pairs into edge indices
        auto q = Axis::Q(A);
        auto r = Axis::R(A);
        std::vector<std::pair<unsigned, unsigned>> ev = {
            {q|r, q|r|A},
            {r, r|A},
            {q, q|A},
            {0, A}};
        for (unsigned i=0; i < 4; ++i)
        {
            es[i] = XTree<3>::mt->e[D ? ev[i].first  : ev[i].second]
                                   [D ? ev[i].second : ev[i].first];
            assert(es[i] != -1);
        }
    }

    uint32_t vs[4];
    for (unsigned i=0; i < ts.size(); ++i)
    {
        // Load either a patch-specific vertex (if this is a lowest-level,
        // potentially non-manifold cell) or the default vertex
        auto vi = ts[i]->level > 0
            ? 0
            : XTree<3>::mt->p[ts[i]->corner_mask][es[i]];
        assert(vi != -1);

        // Sanity-checking manifoldness of collapsed cells
        assert(ts[i]->level == 0 || ts[i]->vertex_count == 1);

        if (ts[i]->index[vi] == 0)
        {
            ts[i]->index[vi] = verts.size();

            verts.push_back(ts[i]->vert(vi).template cast<float>());
        }
        vs[i] = ts[i]->index[vi];
    }

    // Handle polarity-based windings
    if (!D)
    {
        std::swap(vs[1], vs[2]);
    }

    // Pick a triangulation that prevents triangles from folding back
    // on each other by checking normals.
    std::array<Eigen::Vector3f, 4> norms;

    // Computes and saves a corner normal.  a,b,c must be right-handed
    // according to the quad winding, which looks like
    //     2---------3
    //     |         |
    //     |         |
    //     0---------1
    auto saveNorm = [&](int a, int b, int c){
        norms[a] = (verts[vs[b]] - verts[vs[a]]).cross
                   (verts[vs[c]] - verts[vs[a]]).normalized();
    };
    saveNorm(0, 1, 2);
    saveNorm(1, 3, 0);
    saveNorm(2, 0, 3);
    saveNorm(3, 2, 1);
    if (norms[0].dot(norms[3]) > norms[1].dot(norms[2]))
    {
        if (vs[0] != vs[1] && vs[0] != vs[2]) {
            addTriangle({ vs[1], vs[2], vs[0] }, A, D);
            //The order of the triangle matters for negative triangle processing; the one whose diagonal opposite is not included should be last.
        }
        if (vs[3] != vs[1] && vs[3] != vs[2]) {
            addTriangle({ vs[2], vs[1], vs[3] }, A, D);
        }
    }
    else
    {
        if (vs[0] != vs[1] && vs[3] != vs[1]) {
            addTriangle({ vs[3], vs[0], vs[1] }, A, D);
    }
        if (vs[0] != vs[2] && vs[3] != vs[2]) {
            addTriangle({ vs[0], vs[3], vs[2] }, A, D);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<Mesh> Mesh::render(const Tree t, const Region<3>& r,
                                   double min_feature, double max_err)
{
    std::atomic_bool cancel(false);
    std::map<Tree::Id, float> vars;
    return render(t, vars, r, min_feature, max_err, cancel);
}

std::unique_ptr<Mesh> Mesh::render(
            const Tree t, const std::map<Tree::Id, float>& vars,
            const Region<3>& r, double min_feature, double max_err,
            std::atomic_bool& cancel)
{
    // Create the octree (multithreaded and cancellable)
    return mesh(XTree<3>::build(
            t, vars, r, min_feature, max_err, true, cancel), cancel);
}

std::unique_ptr<Mesh> Mesh::render(
        XTreeEvaluator* es,
        const Region<3>& r, double min_feature, double max_err,
        std::atomic_bool& cancel)
{
    return mesh(XTree<3>::build(es, r, min_feature, max_err, true, cancel),
                cancel);
}

std::unique_ptr<Mesh> Mesh::mesh(std::unique_ptr<const XTree<3>> xtree,
                                 std::atomic_bool& cancel)
{
    // Perform marching squares
    auto m = std::unique_ptr<Mesh>(new Mesh());

    if (cancel.load())
    {
        return nullptr;
    }
    else
    {
        Dual<3>::walk(xtree.get(), *m);

#if DEBUG_OCTREE_CELLS
        // Store octree cells as lines
        std::list<const XTree<3>*> todo = {xtree.get()};
        while (todo.size())
        {
            auto t = todo.front();
            todo.pop_front();
            if (t->isBranch())
                for (auto& c : t->children)
                    todo.push_back(c.get());

            static const std::vector<std::pair<uint8_t, uint8_t>> es =
                {{0, Axis::X}, {0, Axis::Y}, {0, Axis::Z},
                 {Axis::X, Axis::X|Axis::Y}, {Axis::X, Axis::X|Axis::Z},
                 {Axis::Y, Axis::Y|Axis::X}, {Axis::Y, Axis::Y|Axis::Z},
                 {Axis::X|Axis::Y, Axis::X|Axis::Y|Axis::Z},
                 {Axis::Z, Axis::Z|Axis::X}, {Axis::Z, Axis::Z|Axis::Y},
                 {Axis::Z|Axis::X, Axis::Z|Axis::X|Axis::Y},
                 {Axis::Z|Axis::Y, Axis::Z|Axis::Y|Axis::X}};
            for (auto e : es)
                m->line(t->cornerPos(e.first).template cast<float>(),
                        t->cornerPos(e.second).template cast<float>());
        }
#endif
        m->processNegativeTriangles();
        m->edgesToBranes.clear();
        return m;
    }
}

void Mesh::processNegativeTriangles() {
    while (!negativeTriangles.empty()) {
        auto triangle = negativeTriangles.back();
        auto removedTriangleIndexLocation = edgesToBranes.find({ triangle.vertices(1), triangle.vertices(0) });
        auto negativeTrianglesSwapped = 0;
        while (removedTriangleIndexLocation == edgesToBranes.end()) {
            negativeTriangle* swappedNegativeTriangle = nullptr;
            for (auto iter = negativeTriangles.begin(); iter != negativeTriangles.end(); ++iter) {
                if (iter->vertices(2) == triangle.vertices(0) && iter->vertices(1) == triangle.vertices(1) ||
                    iter->vertices(2) == triangle.vertices(1) && iter->vertices(0) == triangle.vertices(0)) {
                    swappedNegativeTriangle = &*iter;
                    break;
                }
                else if (iter->vertices(0) == triangle.vertices(1) && iter->vertices(1) == triangle.vertices(0)) {
                    assert(false); //It seems this should not happen; if it does, we've got problems (the negative triangles form a cycle).  Consider throwing an exception.
                    return; //If not in debug mode, return and get some sort of mesh.
                }
            }
            assert(swappedNegativeTriangle);
            std::swap(*swappedNegativeTriangle, negativeTriangles.back());
            triangle = negativeTriangles.back();
            removedTriangleIndexLocation = edgesToBranes.find({ triangle.vertices(1), triangle.vertices(0) });
            ++negativeTrianglesSwapped;
            if (negativeTrianglesSwapped >= negativeTriangles.size()) {
                assert(false); //It seems this should not happen; if it does, we've got problems (the negative triangles form a cycle).  Consider throwing an exception.
                return; //If not in debug mode, return and get some sort of mesh.
            }
        }
        auto removedTriangleIndex = edgesToBranes[{triangle.vertices[1], triangle.vertices[0]}];
        auto removedTriangle = branes[removedTriangleIndex];
        negativeTriangles.pop_back();
        uint32_t removedTriangleExtraPoint;
        if (removedTriangle(0) == triangle.vertices(0)) {
            assert(removedTriangle(2) == triangle.vertices(1));
            removedTriangleExtraPoint = removedTriangle(1);
        }
        else if (removedTriangle(1) == triangle.vertices(0)) {
            assert(removedTriangle(0) == triangle.vertices(1));
            removedTriangleExtraPoint = removedTriangle(2);
        }
        else {
            assert(removedTriangle(2) == triangle.vertices(0));
            assert(removedTriangle(1) == triangle.vertices(1));
            removedTriangleExtraPoint = removedTriangle(0);
        }
        removeTriangle(removedTriangleIndex);
        addTriangle({ removedTriangleExtraPoint, triangle.vertices(1), triangle.vertices(2) }, triangle.A, triangle.D);
        addTriangle({ triangle.vertices(0), removedTriangleExtraPoint, triangle.vertices(2) }, triangle.A, triangle.D);
    }
}

void Mesh::addTriangle(Eigen::Matrix<uint32_t, 3, 1> vertices, Axis::Axis A, bool D) {
    auto scaledOutwardNormal = (verts[vertices(1)] - verts[vertices(0)]).cross(verts[vertices(2)] - verts[vertices(0)]); //No need to normalize here.
    Eigen::Vector3f axisVector({ 0.f, 0.f, 0.f });
    axisVector(Axis::toIndex(A)) = D ? 1.f : -1.f;
    if (scaledOutwardNormal.dot(axisVector) < 0.f) { //We want to make it a negative triangle.
        negativeTriangles.push_back(negativeTriangle{ vertices, A, D });
    }
    else {
        branes.push_back(vertices);
        assert(edgesToBranes.find({ vertices(0), vertices(1) }) == edgesToBranes.end());
        assert(edgesToBranes.find({ vertices(1), vertices(2) }) == edgesToBranes.end());
        assert(edgesToBranes.find({ vertices(2), vertices(0) }) == edgesToBranes.end());
        edgesToBranes[{vertices(0), vertices(1)}] = branes.size() - 1;
        edgesToBranes[{vertices(1), vertices(2)}] = branes.size() - 1;
        edgesToBranes[{vertices(2), vertices(0)}] = branes.size() - 1;
    }
}

void Mesh::removeTriangle(uint32_t target) { //Makes use of the fact that the order doesn't matter to efficiently perform a mid-vector deletion.
    auto removedTriangle = branes[target];
    auto lastTriangle = branes.back();
    assert(edgesToBranes.find({ lastTriangle(0), lastTriangle(1) }) != edgesToBranes.end());
    assert(edgesToBranes.find({ lastTriangle(1), lastTriangle(2) }) != edgesToBranes.end());
    assert(edgesToBranes.find({ lastTriangle(2), lastTriangle(0) }) != edgesToBranes.end());
    edgesToBranes[{lastTriangle(0), lastTriangle(1)}] = target;
    edgesToBranes[{lastTriangle(1), lastTriangle(2)}] = target;
    edgesToBranes[{lastTriangle(2), lastTriangle(0)}] = target;
    edgesToBranes.erase({ removedTriangle(0), removedTriangle(1) });
    edgesToBranes.erase({ removedTriangle(1), removedTriangle(2) });
    edgesToBranes.erase({ removedTriangle(2), removedTriangle(0) });
    branes[target] = branes.back();
    branes.pop_back();
}

void Mesh::line(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
    uint32_t a_ = verts.size();
    verts.push_back(a);
    uint32_t b_ = verts.size();
    verts.push_back(b);

    branes.push_back({a_, a_, b_});
}

////////////////////////////////////////////////////////////////////////////////

bool Mesh::saveSTL(const std::string& filename,
                   const std::list<const Mesh*>& meshes,
                   bool isBinary)
{
  if (!boost::algorithm::iends_with(filename, ".stl"))
  {
    std::cerr << "Mesh::saveSTL: filename \"" << filename
      << "\" does not end in .stl" << std::endl;
    return false;
  }


  if (isBinary)
  {
    FILE * stl_file = fopen(filename.c_str(), "wb");
    if (stl_file == NULL)
    {
      std::cerr << "IOError: " << filename << " could not be opened for writing." << std::endl;
      return false;
    }


    std::string header = "This is a binary STL exported from Ao.";
    // Write unused 80-char header
    for (auto h : header)
    {
      fwrite(&h, sizeof(char), 1, stl_file);
    }

    // Write the rest of the 80-char header
    for (auto h = header.size(); h < 80; h++)
    {
      char o = '_';
      fwrite(&o, sizeof(char), 1, stl_file);
    }

    // Write number of triangles
    unsigned int num_tri = std::accumulate(meshes.begin(), meshes.end(), 0,
                                           [](unsigned int i, const Mesh* m)
                                           {
                                             return i + m->branes.size();
                                           });
    fwrite(&num_tri, sizeof(unsigned int), 1, stl_file);


    for (const auto& m : meshes)
    {
      for (const auto& t : m->branes)
      {
        // Write out the normal vector for this face (all zeros)
        std::vector<float> n(3, 0);
        fwrite(&n[0], sizeof(float), 3, stl_file);

        // Iterate over vertices (which are indices into the verts list)
        for (unsigned i = 0; i < 3; ++i)
        {
          auto vert = m->verts[t[i]];

          std::vector<float> v(3);
          v[0] = vert.x();
          v[1] = vert.y();
          v[2] = vert.z();
          fwrite(&v[0], sizeof(float), 3, stl_file);
        }

        // Write out this face's attribute short
        unsigned short att_count = 0;
        fwrite(&att_count, sizeof(unsigned short), 1, stl_file);
      }
    }

    fclose(stl_file);
  }
  else //ASCII
  {
    auto* stl_file = fopen(filename.c_str(), "w");
    if (stl_file == NULL)
    {
      std::cerr << "IOError: " << filename << " could not be opened for writing." << std::endl;
      return false;
    }
    fprintf(stl_file, "solid %s\n", filename.c_str());

    for (const auto& m : meshes)
    {
      for (const auto& t : m->branes)
      {
        // Write out the normal vector for this face (all zeros)
        fprintf(stl_file, "facet normal ");
        fprintf(stl_file, "0 0 0\n");

        fprintf(stl_file, "outer loop\n");

        // Iterate over vertices (which are indices into the verts list)
        for (unsigned i = 0; i < 3; ++i)
        {
          auto v = m->verts[t[i]];

          fprintf(stl_file,
                  "vertex %e %e %e\n",
                  (float)v.x(),
                  (float)v.y(),
                  (float)v.z());
        }
        fprintf(stl_file, "endloop\n");
        fprintf(stl_file, "endfacet\n");
      }
    }
    fprintf(stl_file, "endsolid %s\n", filename.c_str());
    fclose(stl_file);
  }
  return true;
}

bool Mesh::saveSTL(const std::string& filename,
                   bool isBinary/* = true*/)
{
  return saveSTL(filename, { this }, isBinary);
}

}   // namespace Kernel
