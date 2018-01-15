#include <numeric>
#include <fstream>
#include <boost/algorithm/string/predicate.hpp>

#include "ao/render/brep/mesh.hpp"
#include "ao/render/brep/xtree.hpp"
#include "ao/render/brep/dual.hpp"

namespace Kernel {

template <Axis::Axis A, bool D>
void Mesh::load(const std::array<const XTree<3>*, 4>& ts)
{
    int es[4];
    std::array<int, 2> doubled{ -1, -1 };
    int face = -1; //-x, -y, -z, +x, +y, +z
    Eigen::Vector3f cornerFindingPoint;
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
            es[i] = XTree<3>::mt->e[D ? ev[i].first  : ev[i].second] //first index is filled, second is empty.
                                   [D ? ev[i].second : ev[i].first];
            assert(es[i] != -1);
        }

        //In case of a doubly-used tree, we won't choose the patch from the edge index, but rather choose based on the face and corner.
        if (ts[0] == ts[1]) {
          doubled[0] = 0;
          doubled[1] = 1;
          face = Axis::toIndex(r) + 3;
        }
        else if (ts[0] == ts[2]) {
          doubled[0] = 0;
          doubled[1] = 2;
          face = Axis::toIndex(q) + 3;
        }
        else if (ts[3] == ts[1]) {
          doubled[0] = 3;
          doubled[1] = 1;
          face = Axis::toIndex(q); //The ev's share the absence of q.
        }
        else if (ts[3] == ts[2]) {
          doubled[0] = 3;
          doubled[1] = 2;
          face = Axis::toIndex(r);
        }
        if (doubled[0] != -1) {
          cornerFindingPoint = ts[3 - doubled[0]]->cornerPos(D ? ev[3 - doubled[0]].first : ev[3 - doubled[0]].second).template cast<float>();
            //It may not be on the edge that gave us the quad, but ts[3 - doubled[0]] must be smaller than ts[doubled[0]],
            //and so it will be in the correct quadrant of the face.
        }
    }

    uint32_t vs[4];
    for (unsigned i=0; i < ts.size(); ++i)
    {
        // Load the appropriate vertex
        auto vi = doubled[0] == i || doubled[1] == i ?
           XTree<3>::mt->fp[ts[i]->corner_mask][face] != -2 ?  
            XTree<3>::mt->fp[ts[i]->corner_mask][face] : //Doubled, with a single vertex on this face.
            XTree<3>::mt->cp[ts[i]->corner_mask][ts[i]->closestCorner(cornerFindingPoint)] : //Doubled, with two vertices on this face.
            XTree<3>::mt->p[ts[i]->corner_mask][es[i]]; //Not doubled
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
    for (unsigned i=0; i < norms.size(); ++i)
    {
        norms[i] = (verts[vs[i ^ 1]] - verts[vs[i]]).cross
                   (verts[vs[i ^ 2]] - verts[vs[i]]).normalized();
    }

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
    if (removedTriangle(0) == triangle.vertices(0)){
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
  Eigen::Vector3f axisVector({ 0.f, 0.f, 0.f});
  axisVector(Axis::toIndex(A)) = D ? 1.f : -1.f;
  if (scaledOutwardNormal.dot(axisVector) < 0.f) { //We want to make it a negative triangle.
    negativeTriangles.push_back(negativeTriangle{vertices, A, D});
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

void Mesh::line(Eigen::Vector3f a, Eigen::Vector3f b)
{
    auto a_ = verts.size();
    verts.push_back(a);
    auto b_ = verts.size();
    verts.push_back(b);

    branes.push_back({uint32_t(a_), uint32_t(a_), uint32_t(b_)});
}

////////////////////////////////////////////////////////////////////////////////

bool Mesh::saveSTL(const std::string& filename,
                   const std::list<const Mesh*>& meshes)
{
    if (!boost::algorithm::iends_with(filename, ".stl"))
    {
        std::cerr << "Mesh::saveSTL: filename \"" << filename
                  << "\" does not end in .stl" << std::endl;
    }
    std::ofstream file;
    file.open(filename, std::ios::out);
    if (!file.is_open())
    {
        std::cout << "Mesh::saveSTL: could not open " << filename
                  << std::endl;
        return false;
    }

    // File header (giving human-readable info about file type)
    std::string header = "This is a binary STL exported from Ao.";
    file.write(header.c_str(), header.length());

    // Pad the rest of the header to 80 bytes
    for (int i=header.length(); i < 80; ++i)
    {
        file.put(' ');
    }

    // Write the triangle count to the file
    uint32_t num = std::accumulate(meshes.begin(), meshes.end(), (uint32_t)0,
            [](uint32_t i, const Mesh* m){ return i + m->branes.size(); });
    file.write(reinterpret_cast<char*>(&num), sizeof(num));

    for (const auto& m : meshes)
    {
        for (const auto& t : m->branes)
        {
            // Write out the normal vector for this face (all zeros)
            float norm[3] = {0, 0, 0};
            file.write(reinterpret_cast<char*>(&norm), sizeof(norm));

            // Iterate over vertices (which are indices into the verts list)
            for (unsigned i=0; i < 3; ++i)
            {
                auto v = m->verts[t[i]];
                float vert[3] = {v.x(), v.y(), v.z()};
                file.write(reinterpret_cast<char*>(&vert), sizeof(vert));
            }

            // Write out this face's attribute short
            uint16_t attrib = 0;
            file.write(reinterpret_cast<char*>(&attrib), sizeof(attrib));
        }
    }

    return true;
}

bool Mesh::saveSTL(const std::string& filename)
{
    return saveSTL(filename, {this});
}

}   // namespace Kernel
