#include <numeric>
#include <fstream>
#include <boost/algorithm/string/predicate.hpp>

#include "ao/render/brep/mesh.hpp"
#include "ao/render/brep/xtree.hpp"
#include "ao/render/brep/dual.hpp"

#include "windows.h"

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

            if (ts[i]->index[vi] == 18)
              i = i;

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

    /*for (unsigned i=0; i < norms.size(); ++i)
    {
        norms[i] = (verts[vs[(i + 3) % 4]] - verts[vs[i]]).cross
                   (verts[vs[(i + 1) % 4]] - verts[vs[i]]).normalized();
    }*/

    //  2 3
    //  0 1
    norms[0] = (verts[vs[1]] - verts[vs[0]]).cross
               (verts[vs[2]] - verts[vs[0]]).normalized();
    norms[1] = (verts[vs[0]] - verts[vs[1]]).cross
               (verts[vs[3]] - verts[vs[1]]).normalized();
    norms[2] = (verts[vs[3]] - verts[vs[2]]).cross
               (verts[vs[0]] - verts[vs[2]]).normalized();
    norms[3] = (verts[vs[2]] - verts[vs[3]]).cross
               (verts[vs[1]] - verts[vs[3]]).normalized();

    float dot03 = norms[0].dot(norms[3]), dot12 = norms[1].dot(norms[2]);

    char s[20];

    if (dot03 > dot12)
    {
        branes.push_back({vs[0], vs[1], vs[2]});
        branes.push_back({vs[2], vs[1], vs[3]});

        if (dot03 < -0.9f)
        {
            FILE *f = fopen("quads.txt", "a");
            fprintf(f, "%g %g %g\n", verts[vs[1]][0], verts[vs[1]][1], verts[vs[1]][2]);
            fprintf(f, "%g %g %g\n", verts[vs[3]][0], verts[vs[3]][1], verts[vs[3]][2]);
            fprintf(f, "%g %g %g\n", verts[vs[2]][0], verts[vs[2]][1], verts[vs[2]][2]);
            fprintf(f, "%g %g %g\n", verts[vs[0]][0], verts[vs[0]][1], verts[vs[0]][2]);
            fclose(f);
            char s[100];
            sprintf(s, "xtree IDs: %d, %d, %d, %d\n", ts[0]->id, ts[1]->id, ts[2]->id, ts[3]->id);
            OutputDebugString(s);
        }
    }
    else
    {
        branes.push_back({vs[0], vs[1], vs[3]});
        branes.push_back({vs[0], vs[3], vs[2]});

        if (dot12 < -0.9f)
        {
            FILE *f = fopen("quads.txt", "a");
            fprintf(f, "%g %g %g\n", verts[vs[0]][0], verts[vs[0]][1], verts[vs[0]][2]);
            fprintf(f, "%g %g %g\n", verts[vs[1]][0], verts[vs[1]][1], verts[vs[1]][2]);
            fprintf(f, "%g %g %g\n", verts[vs[3]][0], verts[vs[3]][1], verts[vs[3]][2]);
            fprintf(f, "%g %g %g\n", verts[vs[2]][0], verts[vs[2]][1], verts[vs[2]][2]);
            fclose(f);
            char s[100];
            sprintf(s, "xtree IDs: %d, %d, %d, %d\n", ts[0]->id, ts[1]->id, ts[2]->id, ts[3]->id);
            OutputDebugString(s);
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
        return m;
    }
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
                   const std::list<const Mesh*>& meshes, bool isBinary)
{
    if (!boost::algorithm::iends_with(filename, ".stl"))
    {
        std::cerr << "Mesh::saveSTL: filename \"" << filename
                  << "\" does not end in .stl" << std::endl;
    }

	if (isBinary)
	{
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
		for (int i = header.length(); i < 80; ++i)
		{
			file.put(' ');
		}

		// Write the triangle count to the file
		uint32_t num = std::accumulate(meshes.begin(), meshes.end(), (uint32_t)0,
			[](uint32_t i, const Mesh* m) { return i + m->branes.size(); });
		file.write(reinterpret_cast<char*>(&num), sizeof(num));

		for (const auto& m : meshes)
		{
			for (const auto& t : m->branes)
			{
				// Write out the normal vector for this face (all zeros)
				float norm[3] = { 0, 0, 0 };
				file.write(reinterpret_cast<char*>(&norm), sizeof(norm));

				// Iterate over vertices (which are indices into the verts list)
				for (unsigned i = 0; i < 3; ++i)
				{
					auto v = m->verts[t[i]];
					float vert[3] = { v.x(), v.y(), v.z() };
					file.write(reinterpret_cast<char*>(&vert), sizeof(vert));
				}

				// Write out this face's attribute short
				uint16_t attrib = 0;
				file.write(reinterpret_cast<char*>(&attrib), sizeof(attrib));
			}
		}
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
				auto d1 = m->verts[t[1]] - m->verts[t[0]];
				auto d2 = m->verts[t[2]] - m->verts[t[0]];
				auto c = d1.cross(d2);
				double d = c.dot(c);
//				if (std::abs(d) < 0.000000000000000001)
//					continue;

				// Write out the normal vector for this face (all zeros)
				fprintf(stl_file, "facet normal ");
				fprintf(stl_file, "0 0 0\n");

				fprintf(stl_file, "outer loop\n");

				// Iterate over vertices (which are indices into the verts list)

				for (unsigned i = 0; i < 3; ++i)
				{
					auto v = m->verts[t[i]];

					double x, y, z, B = 1000000000;
					x = std::round(v.x()*B) / B;
					y = std::round(v.y()*B) / B;
					z = std::round(v.z()*B) / B;

					fprintf(stl_file,
						"vertex %f %f %f\n",
						(float)x,
						(float)y,
						(float)z);
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

bool Mesh::saveSTL(const std::string& filename, bool isBinary)
{
	return saveSTL(filename, { this }, isBinary);
}

}   // namespace Kernel
