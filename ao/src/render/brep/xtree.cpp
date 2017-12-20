#include <future>
#include <numeric>
#include <functional>
#include <limits>

#include <Eigen/StdVector>

#include "ao/render/brep/xtree.hpp"
#include "ao/render/axes.hpp"


namespace Kernel {

  static int xtree_id = 0;
  void resetXTreeId() { xtree_id = 0; }
  static double mpf = 0.1; // 10;

//  Here's our cutoff value (with a value set in the header)
template <unsigned N> constexpr double XTree<N>::EIGENVALUE_CUTOFF;

//  Allocating static var for marching cubes table
template <unsigned N>
std::unique_ptr<const Marching::MarchingTable<N>> XTree<N>::mt;

////////////////////////////////////////////////////////////////////////////////

template <unsigned N>
std::unique_ptr<const XTree<N>> XTree<N>::build(
  Tree t, Region<N> region, double min_feature,
  double max_err, bool multithread,
  bool useScope, const PartialOctree* const scope)
{
  std::atomic_bool cancel(false);
  const std::map<Tree::Id, float> vars;
  return build(t, vars, region, min_feature, max_err, multithread, cancel, useScope, scope);
}

template <unsigned N>
std::unique_ptr<const XTree<N>> XTree<N>::build(
  Tree t, const std::map<Tree::Id, float>& vars,
  Region<N> region, double min_feature,
  double max_err, bool multithread,
  std::atomic_bool& cancel,
  bool useScope, const PartialOctree* const scope)
{
  multithread = false;

  if (multithread)
  {
    std::vector<XTreeEvaluator, Eigen::aligned_allocator<XTreeEvaluator>> es;
    es.reserve(1 << N);
    for (unsigned i = 0; i < (1 << N); ++i)
    {
      es.emplace_back(XTreeEvaluator(t, vars));
    }
    return build(es.data(), region, min_feature, max_err, true, cancel, useScope, scope);
  }
  else
  {
    XTreeEvaluator e(t, vars);
    return build(&e, region, min_feature, max_err, false, cancel, useScope, scope);
  }
}

template <unsigned N>
std::unique_ptr<const XTree<N>> XTree<N>::build(
  XTreeEvaluator* es,
  Region<N> region, double min_feature,
  double max_err, bool multithread,
  std::atomic_bool& cancel,
  bool useScope, const PartialOctree* const scope)
{
  // Lazy initialization of marching squares / cubes table
  if (mt.get() == nullptr)
  {
    mt = Marching::buildTable<N>();
  }

  auto out = new XTree(es, region, min_feature, max_err, multithread, cancel);

  // Return an empty XTree when cancelled
  // (to avoid potentially ambiguous or mal-constructed trees situations)
  if (cancel.load())
  {
    delete out;
    out = nullptr;
  }

  return std::unique_ptr<const XTree<N>>(out);
}

////////////////////////////////////////////////////////////////////////////////

template <unsigned N>
XTree<N>::XTree(XTreeEvaluator* eval, Region<N> region,
  double min_feature, double max_err, bool multithread,
  std::atomic_bool& cancel, bool useScope = false, const PartialOctree* const scope = nullptr)
  : region(region), _mass_point(Eigen::Matrix<double, N + 1, 1>::Zero()),
  AtA(Eigen::Matrix<double, N, N>::Zero()),
  AtB(Eigen::Matrix<double, N, 1>::Zero())
{
  id = xtree_id++;

  if (id == 22082)
  {
    bool isLeaf = !isBranch();
    FILE *f = fopen("data.txt", "a");
    fprintf(f, "*** id = %d is %s a leaf\n", id, isLeaf ? "SO" : "NOT");
    fclose(f);
  }

  if (                       ((id % 10000) == 0) 
      //|| (id >= 3380000 && ((id % 1000) == 0))
      //|| (id >= 3389000 && ((id % 100) == 0))
      //|| (id >= 3389600)
    )
  {
    //char s[20];
    //sprintf(s, "ids%g.txt", min_feature);
    //OutputDebugString(s);
    FILE *f = fopen("ids.txt", "a");
    fprintf(f, " %d \n", id);
    fclose(f);
  }

  if (id == 3389602)
    id = id;

  if (cancel.load())
  {
    return;
  }

  if (id == 3389602)
  {
    FILE *f = fopen("ids.txt", "a");
    fprintf(f, "129\n");
    fclose(f);
  }

  if (useScope && scope == nullptr)
  {
    type = Interval::OUTOFSCOPE;
    std::fill(corners.begin(), corners.end(), Interval::OUTOFSCOPE);

    if (id == 3389602)
    {
      FILE *f = fopen("ids.txt", "a");
      fprintf(f, "141\n");
      fclose(f);
    }

    return; //This node of the tree won't be used, so nothing more needs to happen.
  }

  if (id == 3389602)
  {
    FILE *f = fopen("ids.txt", "a");
    fprintf(f, "151\n");
    fclose(f);
  }

  // Clear all indices
  std::fill(index.begin(), index.end(), 0);

  if (id == 3389602)
  {
    FILE *f = fopen("ids.txt", "a");
    fprintf(f, "161\n");
    fclose(f);
  }

  // Do a preliminary evaluation to prune the tree
  auto i = eval->interval.evalAndPush(region.lower3().template cast<float>(),
    region.upper3().template cast<float>());

  if (id == 3389602)
  {
    FILE *f = fopen("ids.txt", "a");
    fprintf(f, "172\n");
    fclose(f);
  }

  if (Interval::isFilled(i))
  {
    if (id == 3389602)
    {
      FILE *f = fopen("ids.txt", "a");
      fprintf(f, "181\n");
      fclose(f);
    }
    type = Interval::FILLED;
  }
  else if (Interval::isEmpty(i))
  {
    if (id == 3389602)
    {
      FILE *f = fopen("ids.txt", "a");
      fprintf(f, "191\n");
      fclose(f);
    }
    type = Interval::EMPTY;
  }
  // If the cell wasn't empty or filled, attempt to subdivide and recurse
  else
  {
    if (id == 3389602)
    {
      FILE *f = fopen("ids.txt", "a");
      fprintf(f, "202\n");
      fclose(f);
    }
    bool all_empty = true;
    bool all_full = true;

    // Recurse until volume is too small
    if (((region.upper - region.lower) > min_feature).any())
    {
      if (id == 3389602)
      {
        FILE *f = fopen("ids.txt", "a");
        fprintf(f, "214\n");
        fclose(f);
      }
      auto rs = region.subdivide();

      multithread = false;

      if (multithread)
      {
        // Evaluate every child in a separate thread
        std::array<std::future<XTree<N>*>, 1 << N> futures;

        assert(children.size() == futures.size());

        for (unsigned i = 0; i < children.size(); ++i)
        {
          futures[i] = std::async(std::launch::async,
            [&eval, &rs, i, min_feature, max_err, &cancel, &useScope, &scope]()
          { return new XTree(
            eval + i, rs[i], min_feature, max_err,
            false, cancel, useScope, useScope ? scope->children[i].get() : nullptr); });
        }
        for (unsigned i = 0; i < children.size(); ++i)
        {
          children[i].reset(futures[i].get());
        }
      }
      // Single-threaded recursive construction
      else
      {
        for (uint8_t i = 0; i < children.size(); ++i)
        {
          // Populate child recursively
          children[i].reset(new XTree<N>(
            eval, rs[i], min_feature, max_err,
            false, cancel, useScope ? scope->children[i].get() : nullptr));
        }
      }

      // Abort early if children could have been mal-constructed
      // by an early cancel operation
      if (cancel.load())
      {
        eval->interval.pop();
        return;
      }

      if (id == 3389602)
      {
        FILE *f = fopen("ids.txt", "a");
        fprintf(f, "240\n");
        fclose(f);
      }

      // Update corner and filled / empty state from children
      for (uint8_t i = 0; i < children.size(); ++i)
      {
        // Grab corner values from children
        corners[i] = children[i]->corners[i];
        all_empty &= children[i]->type == Interval::EMPTY || children[i]->type == Interval::OUTOFSCOPE;
        all_full &= children[i]->type == Interval::FILLED || children[i]->type == Interval::OUTOFSCOPE;
      }
    }
    // Terminate recursion here
    else
    {
      if (id == 3389602)
      {
        FILE *f = fopen("ids.txt", "a");
        fprintf(f, "283\n");
        fclose(f);
      }
      // Pack corners into evaluator
      std::array<Eigen::Vector3f, 1 << N> pos;
      for (uint8_t i = 0; i < children.size(); ++i)
      {
        if (id == 3389602) { FILE *f = fopen("ids.txt", "a"); fprintf(f, "290: i = %d\n", i); fclose(f); }
        pos[i] << cornerPos(i).template cast<float>(),
          region.perp.template cast<float>();
        if (id == 3389602) { FILE *f = fopen("ids.txt", "a"); fprintf(f, "293\n"); fclose(f); }
        eval->array.set(pos[i], i);
        if (id == 3389602) { FILE *f = fopen("ids.txt", "a"); fprintf(f, "295\n"); fclose(f); }
      }

      if (id == 3389602) { FILE *f = fopen("ids.txt", "a"); fprintf(f, "295\n"); fclose(f); }

      // Evaluate the region's corners and check their states
      auto ds = eval->array.derivs(children.size());
      auto ambig = eval->array.getAmbiguous(children.size());
      for (uint8_t i = 0; i < children.size(); ++i)
      {
        // Handle inside, outside, and (non-ambiguous) on-boundary
        if (ds.col(i).w() < 0) { corners[i] = Interval::FILLED; }
        else if (ds.col(i).w() > 0) { corners[i] = Interval::EMPTY; }
        else if (!ambig(i))
        {
          // Optimization for non-ambiguous features
          // (see explanation in FeatureEvaluator::isInside)
          corners[i] = (ds.col(i).template head<3>() != 0).any()
            ? Interval::FILLED : Interval::EMPTY;
        }
      }

      if (id == 3389602) { FILE *f = fopen("ids.txt", "a"); fprintf(f, "314\n"); fclose(f); }

      // Separate pass for handling ambiguous corners
      for (uint8_t i = 0; i < children.size(); ++i)
      {
        if (id == -1) //540) //84267816)
          i = i;
        if (ds.col(i).w() == 0 && ambig(i))
        {
          corners[i] = eval->feature.isInside(pos[i])
            ? Interval::FILLED : Interval::EMPTY;
        }
      }

      if (id == 3389602) { FILE *f = fopen("ids.txt", "a"); fprintf(f, "328\n"); fclose(f); }

      if (id == 22082) //540) //84267816)
      {
        FILE *f = fopen("data.txt", "a");
        Eigen::Vector3f black(0, 0, 0), white(1, 1, 1);
        Eigen::Vector3f diag = (pos[7] - pos[0]) / 60, ctr = (pos[0] + pos[7]) / 2;
        for (uint8_t i = 0; i < children.size(); ++i)
        {
          dbgAddBox(pos[i] - diag, pos[i] + diag, (corners[i] == Interval::FILLED) ? black : white);
          fprintf(f, "%d: %d   (%g %g %g)\n", i, corners[i], pos[i][0], pos[i][1], pos[i][2]);
          Eigen::Vector3f incr = (pos[i] - ctr) / 15;
          for (int ii = 1; ii <= i; ii++)
            dbgAddPoint(pos[i] + ii * incr, black);
        }
        fprintf(f, "------\n");
        fclose(f);
      }

      // Pack corners into filled / empty arrays
      for (uint8_t i = 0; i < children.size(); ++i)
      {
        all_full &= corners[i];
        all_empty &= !corners[i];
      }

      if (id == 3389602) { FILE *f = fopen("ids.txt", "a"); fprintf(f, "295\n"); fclose(f); }
    }

    if (id == 3389602)
    {
      FILE *f = fopen("ids.txt", "a");
      fprintf(f, "352\n");
      fclose(f);
    }
    assert(!all_empty || !all_full); //If it's both, that means all children were out of scope, meaning that the scope 
      //PartialOctree did not extend to the proper depth.
    type = all_empty ? Interval::EMPTY
      : all_full ? Interval::FILLED : Interval::AMBIGUOUS;
  }

  if (id == 3389602)
  {
    FILE *f = fopen("ids.txt", "a");
    fprintf(f, "345\n");
    fclose(f);
  }

  // If this cell is unambiguous, then fill its corners with values and
  // forget all its branches; these may be no-ops, but they're idempotent
  if (type == Interval::FILLED || type == Interval::EMPTY)
  {
    std::fill(corners.begin(), corners.end(), type);
    std::for_each(children.begin(), children.end(),
      [](std::unique_ptr<const XTree<N>>& o) { o.reset(); });
    manifold = true;
  }

  // Build and store the corner mask
  for (unsigned i = 0; i < children.size(); ++i)
  {
    corner_mask |= (corners[i] == Interval::FILLED) << i;
  }

  // Branch checking and simplifications
  if (isBranch())
  {
    // Store this tree's depth as a function of its children
    level = std::accumulate(children.begin(), children.end(), (unsigned)0,
      [](const unsigned& a, const std::unique_ptr<const XTree<N>>& b)
    { return std::max(a, b->level); }) + 1;

    // If all children are non-branches, then we could collapse
    if (std::all_of(children.begin(), children.end(),
      [](const std::unique_ptr<const XTree<N>>& o)
    { return !o->isBranch(); }))
    {
      //  This conditional implements the three checks described in
      //  [Ju et al, 2002] in the section titled
      //      "Simplification with topology safety"
      manifold = cornersAreManifold() &&
        std::all_of(children.begin(), children.end(),
          [](const std::unique_ptr<const XTree<N>>& o)
      { return o->manifold; }) &&
        leafsAreManifold();

      // Attempt to collapse this tree by positioning the vertex
      // in the summed QEF and checking to see if the error is small
      if (manifold)
      {
        // Populate the feature rank as the maximum of all children
        // feature ranks (as seen in DC: The Secret Sauce)
        rank = std::accumulate(
          children.begin(), children.end(), (unsigned)0,
          [](unsigned a, const std::unique_ptr<const XTree<N>>& b)
        { return std::max(a, b->rank); });

        // Accumulate the mass point and QEF matrices
        for (const auto& c : children)
        {
          if (c->rank == rank)
          {
            _mass_point += c->_mass_point;
          }
          AtA += c->AtA;
          AtB += c->AtB;
          BtB += c->BtB;
        }
        assert(region.contains(massPoint()));

        // If the vertex error is below a threshold, and the vertex
        // is well-placed in the distance field, then convert into
        // a leaf by erasing all of the child branches
        if (findVertex(vertex_count++) < max_err &&
          fabs(eval->feature.baseEval(vert3().template cast<float>()))
          < max_err)
        {
          std::for_each(children.begin(), children.end(),
            [](std::unique_ptr<const XTree<N>>& o) { o.reset(); });
        }
        else
        {
          vertex_count = 0;
        }
      }
    }
  }
  else if (type == Interval::AMBIGUOUS)
  {
    // Figure out if the leaf is manifold
    manifold = cornersAreManifold();

    // Here, we'll prepare to store position, {normal, value} pairs
    // for every crossing and feature
    typedef std::pair<Vec, Eigen::Matrix<double, N + 1, 1>> Intersection;
    std::vector<Intersection, Eigen::aligned_allocator<Intersection>>
      intersections;

    // RAM is cheap, so reserve a bunch of space here to avoid
    // the need for re-allocating later on
    intersections.reserve(_edges(N) * 2);

    // We'll use this vector anytime we need to pass something
    // into the evaluator (which requires a Vector3f)
    Eigen::Vector3f _pos;
    _pos.template tail<3 - N>() = region.perp.template cast<float>();
    auto set = [&](const Vec& v, size_t i) {
      _pos.template head<N>() = v.template cast<float>();
      eval->array.set(_pos, i);
    };

    // Iterate over manifold patches for this corner case
    const auto& ps = mt->v[corner_mask];
    while (vertex_count < ps.size() && ps[vertex_count][0].first != -1)
    {
      // Iterate over edges in this patch, storing [inside, outside]
      unsigned target_count;
      std::array<std::pair<Vec, Vec>, _edges(N)> targets;
      std::array<int, _edges(N)> ranks;
      std::fill(ranks.begin(), ranks.end(), -1);

      for (target_count = 0; target_count < ps[vertex_count].size() &&
        ps[vertex_count][target_count].first != -1; ++target_count)
      {
        // Sanity-checking
        assert(corners[ps[vertex_count][target_count].first]
          == Interval::FILLED);
        assert(corners[ps[vertex_count][target_count].second]
          == Interval::EMPTY);

        // Store inside / outside in targets array
        targets[target_count] = { cornerPos(ps[vertex_count][target_count].first),
          cornerPos(ps[vertex_count][target_count].second) };
      }

      // We do an N-fold reduction at each stage
      constexpr int SEARCH_COUNT = 4;
      constexpr int POINTS_PER_SEARCH = 16;
      static_assert(_edges(N) * POINTS_PER_SEARCH <= ArrayEvaluator::N,
        "Potential overflow");

      // Multi-stage binary search for intersection
      for (int s = 0; s < SEARCH_COUNT; ++s)
      {
        // Load search points into evaluator
        Eigen::Array<double, N, POINTS_PER_SEARCH * _edges(N)> ps;
        for (unsigned e = 0; e < target_count; ++e)
        {
          for (int j = 0; j < POINTS_PER_SEARCH; ++j)
          {
            const double frac = j / (POINTS_PER_SEARCH - 1.0);
            const unsigned i = j + e*POINTS_PER_SEARCH;
            ps.col(i) = (targets[e].first * (1 - frac)) +
              (targets[e].second * frac);
            set(ps.col(i), i);
          }
        }

        // Evaluate, then search for the first outside point
        // and adjust inside / outside to their new positions
        auto out = eval->array.values(
          POINTS_PER_SEARCH * target_count);

        for (unsigned e = 0; e < target_count; ++e)
        {
          for (unsigned j = 0; j < POINTS_PER_SEARCH; ++j)
          {
            const unsigned i = j + e*POINTS_PER_SEARCH;
            if (out[i] > 0)
            {
              targets[e] = { ps.col(i - 1), ps.col(i) };
              break;
            }
            else if (out[i] == 0)
            {
              Eigen::Vector3d pos;
              pos << ps.col(i), region.perp;
              if (!eval->feature.isInside(
                pos.template cast<float>()))
              {
                targets[e] = { ps.col(i - 1), ps.col(i) };
                break;
              }
            }
          }
        }
      }

      // Reset mass point and intersections
      _mass_point = _mass_point.Zero();
      intersections.clear();

      // Store mass point and prepare for a bulk evaluation
      static_assert(_edges(N) * 2 <= ArrayEvaluator::N,
        "Too many results");
      for (unsigned i = 0; i < target_count; ++i)
      {
        set(targets[i].first, 2 * i);
        set(targets[i].second, 2 * i + 1);
      }

      // Evaluate values and derivatives
      auto ds = eval->array.derivs(2 * target_count);
      auto ambig = eval->array.getAmbiguous(2 * target_count);

      // Iterate over all inside-outside pairs, storing the number
      // of intersections before each inside node (in prev_size), then
      // checking the rank of the pair after each outside node based
      // on the accumulated intersections.
      size_t prev_size;
      for (unsigned i = 0; i < 2 * target_count; ++i)
      {
        if (!(i & 1))
        {
          prev_size = intersections.size();
        }

        if (!ambig(i))
        {
          const Eigen::Array<double, N, 1> derivs = ds.col(i)
            .template cast<double>().template head<N>();
          const double norm = derivs.matrix().norm();

          // Find normalized derivatives and distance value
          Eigen::Matrix<double, N + 1, 1> dv;
          dv << derivs / norm, ds.col(i).w() / norm;
          if (!dv.array().isNaN().any())
          {
            intersections.push_back({
              (i & 1) ? targets[i / 2].second : targets[i / 2].first,
              dv });
          }
        }
        else
        {
          // Load the ambiguous position and find its features
          Eigen::Vector3f pos;
          pos << ((i & 1) ? targets[i / 2].second : targets[i / 2].first)
            .template cast<float>(),
            region.perp.template cast<float>();
          const auto fs = eval->feature.featuresAt(pos);

          for (auto& f : fs)
          {
            // Evaluate feature-specific distance and
            // derivatives value at this particular point
            eval->feature.push(f);

            const auto ds = eval->feature.deriv(pos, f);
                //If we hadn't loaded the derivatives for that point during featuresAt, 
                //we'd need to do so first.

            // Unpack 3D derivatives into XTree-specific
            // dimensionality, and find normal.
            const Eigen::Array<double, N, 1> derivs = ds
              .template head<N>()
              .template cast<double>();
            const double norm = derivs.matrix().norm();

            // Find normalized derivatives and distance value
            Eigen::Matrix<double, N + 1, 1> dv;
            dv << derivs / norm, ds.w() / norm;
            if (!dv.array().isNaN().any())
            {
              intersections.push_back({
                (i & 1) ? targets[i / 2].second
                : targets[i / 2].first, dv });
            }
            eval->feature.pop();
          }
        }

        if (i & 1)
        {
          // If every intersection was NaN (?!), use rank 0;
          // otherwise, figure out how many normals diverge
          if (prev_size == intersections.size())
          {
            ranks[i / 2] = 0;
          }
          else
          {
            ranks[i / 2] = 1;
            Vec prev_normal = intersections[prev_size]
              .second
              .template head<N>();
            for (unsigned p = prev_size + 1;
              p < intersections.size(); ++p)
            {
              // Accumulate rank based on cosine distance
              ranks[i / 2] += intersections[p]
                .second
                .template head<N>()
                .dot(prev_normal) < 0.9;
            }
          }
        }
      }

      {   // Build the mass point from max-rank intersections
        const int max_rank = *std::max_element(
          ranks.begin(), ranks.end());
        for (unsigned i = 0; i < target_count; ++i)
        {
          assert(ranks[i] != -1);
          if (ranks[i] == max_rank)
          {
            // Accumulate this intersection in the mass point
            Eigen::Matrix<double, N + 1, 1> mp;
            mp << targets[i].first, 1;
            _mass_point += mp;
            mp << targets[i].second, 1;
            _mass_point += mp;
          }
        }
      }

      // Now, we'll unpack into A and b matrices
      //
      //  The A matrix is of the form
      //  [n1x, n1y, n1z]
      //  [n2x, n2y, n2z]
      //  [n3x, n3y, n3z]
      //  ...
      //  (with one row for each sampled point's normal)
      Eigen::Matrix<double, Eigen::Dynamic, N> A(intersections.size(), N);

      //  The b matrix is of the form
      //  [p1 . n1]
      //  [p2 . n2]
      //  [p3 . n3]
      //  ...
      //  (with one row for each sampled point)
      Eigen::Matrix<double, Eigen::Dynamic, 1> b(intersections.size(), 1);

      // Load samples into the QEF arrays
      //
      // Since we're deliberately sampling on either side of the
      // intersection, we subtract out the distance-field value
      // to make the math work out.
      for (unsigned i = 0; i < intersections.size(); ++i)
      {
        A.row(i) << intersections[i].second
          .template head<N>()
          .transpose();
        b(i) = A.row(i).dot(intersections[i].first) -
          intersections[i].second(N);
      }

      if (id == 22082) //540) //84267816)
      {
        Eigen::Vector3f L, U, p, n, ppn, red(1, 0, 0), yellow(1, 1, 0), purple(1, 0, 1), diff, near;
        for (int ii = 0; ii < 3; ii++)
        {
          L[ii] = region.lower[ii];
          U[ii] = region.upper[ii];
        }
        dbgAddBox(L, U, yellow);
        FILE *f = fopen("data.txt", "a");
        for (unsigned i = 0; i < intersections.size(); ++i)
        {
          for (int ii = 0; ii < 3; ii++)
          {
            p[ii] = intersections[i].first[ii];
            n[ii] = intersections[i].second[ii];
            ppn[ii] = p[ii] + n[ii];
          }
          double nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
          assert(nn > 0.999 && nn < 1.001);
          dbgAddPoint(p, red);
          dbgAddLine(p, ppn, purple);
          fprintf(f, "(%g %g %g)   (%g %g %g)\n", intersections[i].first[0], intersections[i].first[1], intersections[i].first[2],
            intersections[i].second[0], intersections[i].second[1], intersections[i].second[2]);
          for (int ii = 0; ii < 3; ii++)
            data.push_back(p[ii]);
        }
        fclose(f);
      }

      // Save compact QEF matrices

      auto center = massPoint();

      auto At = A.transpose().eval();
      AtA = At * A;
      AtB = At * b;
      BtB = b.transpose() * b;

      /*AtA(0, 0) += mpf;
      AtA(1, 1) += mpf;
      AtA(2, 2) += mpf;
      AtB += mpf * center;
      BtB += mpf * center.transpose() * center;*/

      // Find the vertex position, storing into the appropriate column
      // of the vertex array and ignoring the error result (because
      // this is the bottom of the recursion)
      findVertex(vertex_count++);
    }
  }

  // ...and we're done.
  eval->interval.pop();

  if (id == 3389602)
  {
    FILE *f = fopen("ids.txt", "a");
    fprintf(f, "723\n");
    fclose(f);
  }
}


////////////////////////////////////////////////////////////////////////////////

template <unsigned N>
typename XTree<N>::Vec XTree<N>::correctVertexPosition(Vec lower, Vec upper, Vec v)
{
  Vec center = (lower + upper) / 2, vv;
  double q, eps = 0.000000001;
  int i, j, k;

  for (i = 0, j = 1, k = 2; i < 3; i++, j = (j + 1) % 3, k = (k + 1) % 3)
  {
    if (v(i) < lower(i))
    {
      q = (lower(i) - v(i)) / (center(i) - v(i));
      vv = q * center + (1 - q) * v;
      if (vv(j) > lower(j) - eps && vv(j) < upper(j) + eps && vv(k) > lower(k) - eps && vv(k) < upper(k) + eps)
        return vv;
    }
  }
  for (i = 0, j = 1, k = 2; i < 3; i++, j = (j + 1) % 3, k = (k + 1) % 3)
  {
    if (v(i) > upper(i))
    {
      q = (v(i) - upper(i)) / (v(i) - center(i));
      vv = q * center + (1 - q) * v;
      if (vv(j) > lower(j) - eps && vv(j) < upper(j) + eps && vv(k) > lower(k) - eps && vv(k) < upper(k) + eps)
        return vv;
    }
  }
  return v;
}

static double qMax = 0.0;
static int qMaxId = -1;

template <unsigned N>
void XTree<N>::maxOutBy(double &max, int &maxId)
{
  max = qMax;
  maxId = qMaxId;
}

template <unsigned N>
void XTree<N>::outBy(Vec lower, Vec upper, Vec v)
{
  int i, j, k;
  double q, max_q = 0.0;

  for (i = 0, j = 1, k = 2; i < 3; i++, j = (j + 1) % 3, k = (k + 1) % 3)
  {
    if (v(i) < lower(i))
    {
      q = (lower(i) - v(i)) / (upper(i) - lower(i));
      if (q > max_q)
        max_q = q;
    }
  }
  for (i = 0, j = 1, k = 2; i < 3; i++, j = (j + 1) % 3, k = (k + 1) % 3)
  {
    if (v(i) > upper(i))
    {
      q = (v(i) - upper(i)) / (upper(i) - lower(i));
      if (q > max_q)
        max_q = q;
    }
  }

  if (max_q > qMax)
  {
    qMax = max_q;
    qMaxId = id;
  }
}

template <unsigned N>
double XTree<N>::findVertex(unsigned index)
{
  Eigen::EigenSolver<Eigen::Matrix<double, N, N>> es(AtA);
  assert(_mass_point(N) > 0);

  // We need to find the pseudo-inverse of AtA.
  auto eigenvalues = es.eigenvalues().real();
  double cutoff = EIGENVALUE_CUTOFF;
  Vec v;

  while (true)
  {
    // Truncate near-singular eigenvalues in the SVD's diagonal matrix
    Eigen::Matrix<double, N, N> D = Eigen::Matrix<double, N, N>::Zero();
    for (unsigned i = 0; i < N; ++i)
    {
      D.diagonal()[i] = (std::abs(eigenvalues[i]) < cutoff)
        ? 0 : (1 / eigenvalues[i]);
    }

    // Get rank from eigenvalues
    auto rank_ = rank;
    if (!isBranch())
    {
      assert(index > 0 || rank == 0);
      rank = D.diagonal().count();
    }

    // SVD matrices
    auto U = es.eigenvectors().real().eval(); // = V

                          // Pseudo-inverse of A
    auto AtAp = (U * D * U.transpose()).eval();

    // Solve for vertex position (minimizing distance to center)
    auto center = massPoint();
    v = AtAp * (AtB - (AtA * center)) + center;

    if (id == 22082) //540) //84267816)
    {
      Eigen::Vector3f p, red(1, 0, 0);
      p[0] = v[0]; p[1] = v[1]; p[2] = v[2];
      dbgAddPoint(p, red);
    }

    if (v(0) >= region.lower(0) && v(1) >= region.lower(1) && v(2) >= region.lower(2) &&
        v(0) <= region.upper(0) && v(1) <= region.upper(1) && v(2) <= region.upper(2))
      break;

    //v = massPoint();
    //break;

    //v = correctVertexPosition(region.lower, region.upper, v);
    if (!isBranch())
    {
      /*FILE *f = fopen("data.txt", "a");
      fprintf(f, "*** id = %d\n", id);
      fclose(f);*/
      outBy(region.lower, region.upper, v);
    }
    break;

    if (rank == 0)
    {
      //assert(false);
      // v = (region.lower + region.upper) / 2;
      FILE *f_ = fopen("cutoffs.txt", "a");
      fprintf(f_, "tree %d, index %d: failed even for cutoff %g\n", id, index, cutoff);
      fclose(f_);
      break;
    }
    double cutoff_ = 1000000000.0;
    for (unsigned i = 0; i < N; ++i)
    {
      if (std::abs(eigenvalues[i]) < cutoff)
        continue;
      if (std::abs(eigenvalues[i]) < cutoff_)
        cutoff_ = std::abs(eigenvalues[i]) + 0.000000001;
    }
    cutoff = cutoff_;
    rank = rank_;
    FILE *f = fopen("cutoffs.txt", "a");
    fprintf(f, "tree %d, index %d: trying cutoff = %g\n", id, index, cutoff);
    fclose(f);
  }

  // Store this specific vertex in the verts matrix
  verts.col(index) = v;

  // Return the QEF error
  return (v.transpose() * AtA * v - 2 * v.transpose() * AtB)[0] + BtB;
}

////////////////////////////////////////////////////////////////////////////////

template <unsigned N>
Eigen::Vector3d XTree<N>::vert3(unsigned index) const
{
  Eigen::Vector3d out;
  out << vert(index), region.perp.template cast<double>();
  return out;
}

template <unsigned N>
typename XTree<N>::Vec XTree<N>::massPoint() const
{
  return _mass_point.template head<N>() / _mass_point(N);
}

////////////////////////////////////////////////////////////////////////////////
// Specializations for quadtree
template <>
bool XTree<2>::leafsAreManifold() const
{
  /*  See detailed comment in Octree::leafTopology */
  const bool edges_safe =
    (child(0)->cornerState(Axis::X) == cornerState(0) ||
      child(0)->cornerState(Axis::X) == cornerState(Axis::X))
    && (child(0)->cornerState(Axis::Y) == cornerState(0) ||
      child(0)->cornerState(Axis::Y) == cornerState(Axis::Y))
    && (child(Axis::X)->cornerState(Axis::X | Axis::Y) == cornerState(Axis::X) ||
      child(Axis::X)->cornerState(Axis::X | Axis::Y) == cornerState(Axis::X | Axis::Y))
    && (child(Axis::Y)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::Y) ||
      child(Axis::Y)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::Y | Axis::X));

  const bool faces_safe =
    (child(0)->cornerState(Axis::Y | Axis::X) == cornerState(0) ||
      child(0)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::Y) ||
      child(0)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::X) ||
      child(0)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::Y | Axis::X));

  return edges_safe && faces_safe;
}

template <>
bool XTree<2>::cornersAreManifold() const
{
  const static bool corner_table[] =
  { 1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1 };
  return corner_table[corner_mask];
}

template <>
const std::vector<std::pair<uint8_t, uint8_t>>& XTree<2>::edges() const
{
  static const std::vector<std::pair<uint8_t, uint8_t>> es =
  { { 0, Axis::X },{ 0, Axis::Y },
  { Axis::X, Axis::X | Axis::Y },{ Axis::Y, Axis::Y | Axis::X } };
  return es;
}

////////////////////////////////////////////////////////////////////////////////
// Specializations for octree
template <>
bool XTree<3>::leafsAreManifold() const
{
  /*  - The sign in the middle of a coarse edge must agree with the sign of at
  *    least one of the edge’s two endpoints.
  *  - The sign in the middle of a coarse face must agree with the sign of at
  *    least one of the face’s four corners.
  *  - The sign in the middle of a coarse cube must agree with the sign of at
  *    least one of the cube’s eight corners.
  *  [Ju et al, 2002]    */

  // Check the signs in the middle of leaf cell edges
  const bool edges_safe =
    (child(0)->cornerState(Axis::Z) == cornerState(0) ||
      child(0)->cornerState(Axis::Z) == cornerState(Axis::Z))
    && (child(0)->cornerState(Axis::X) == cornerState(0) ||
      child(0)->cornerState(Axis::X) == cornerState(Axis::X))
    && (child(0)->cornerState(Axis::Y) == cornerState(0) ||
      child(0)->cornerState(Axis::Y) == cornerState(Axis::Y))

    && (child(Axis::X)->cornerState(Axis::X | Axis::Y) == cornerState(Axis::X) ||
      child(Axis::X)->cornerState(Axis::X | Axis::Y) == cornerState(Axis::X | Axis::Y))
    && (child(Axis::X)->cornerState(Axis::X | Axis::Z) == cornerState(Axis::X) ||
      child(Axis::X)->cornerState(Axis::X | Axis::Z) == cornerState(Axis::X | Axis::Z))

    && (child(Axis::Y)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::Y) ||
      child(Axis::Y)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::Y | Axis::X))
    && (child(Axis::Y)->cornerState(Axis::Y | Axis::Z) == cornerState(Axis::Y) ||
      child(Axis::Y)->cornerState(Axis::Y | Axis::Z) == cornerState(Axis::Y | Axis::Z))

    && (child(Axis::X | Axis::Y)->cornerState(Axis::X | Axis::Y | Axis::Z) ==
      cornerState(Axis::X | Axis::Y) ||
      child(Axis::X | Axis::Y)->cornerState(Axis::X | Axis::Y | Axis::Z) ==
      cornerState(Axis::X | Axis::Y | Axis::Z))

    && (child(Axis::Z)->cornerState(Axis::Z | Axis::X) == cornerState(Axis::Z) ||
      child(Axis::Z)->cornerState(Axis::Z | Axis::X) == cornerState(Axis::Z | Axis::X))
    && (child(Axis::Z)->cornerState(Axis::Z | Axis::Y) == cornerState(Axis::Z) ||
      child(Axis::Z)->cornerState(Axis::Z | Axis::Y) == cornerState(Axis::Z | Axis::Y))

    && (child(Axis::Z | Axis::X)->cornerState(Axis::Z | Axis::X | Axis::Y) ==
      cornerState(Axis::Z | Axis::X) ||
      child(Axis::Z | Axis::X)->cornerState(Axis::Z | Axis::X | Axis::Y) ==
      cornerState(Axis::Z | Axis::X | Axis::Y))

    && (child(Axis::Z | Axis::Y)->cornerState(Axis::Z | Axis::Y | Axis::X) ==
      cornerState(Axis::Z | Axis::Y) ||
      child(Axis::Z | Axis::Y)->cornerState(Axis::Z | Axis::Y | Axis::X) ==
      cornerState(Axis::Z | Axis::Y | Axis::X));

  const bool faces_safe =
    (child(0)->cornerState(Axis::X | Axis::Z) == cornerState(0) ||
      child(0)->cornerState(Axis::X | Axis::Z) == cornerState(Axis::X) ||
      child(0)->cornerState(Axis::X | Axis::Z) == cornerState(Axis::Z) ||
      child(0)->cornerState(Axis::X | Axis::Z) == cornerState(Axis::X | Axis::Z))
    && (child(0)->cornerState(Axis::Y | Axis::Z) == cornerState(0) ||
      child(0)->cornerState(Axis::Y | Axis::Z) == cornerState(Axis::Y) ||
      child(0)->cornerState(Axis::Y | Axis::Z) == cornerState(Axis::Z) ||
      child(0)->cornerState(Axis::Y | Axis::Z) == cornerState(Axis::Y | Axis::Z))
    && (child(0)->cornerState(Axis::Y | Axis::X) == cornerState(0) ||
      child(0)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::Y) ||
      child(0)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::X) ||
      child(0)->cornerState(Axis::Y | Axis::X) == cornerState(Axis::Y | Axis::X))

    && (child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::X) == cornerState(Axis::X) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::X) == cornerState(Axis::X | Axis::Z) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::X) == cornerState(Axis::X | Axis::Y) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::X) ==
      cornerState(Axis::X | Axis::Y | Axis::Z))
    && (child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::Y) == cornerState(Axis::Y) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::Y) == cornerState(Axis::Y | Axis::Z) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::Y) == cornerState(Axis::Y | Axis::X) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::Y) ==
      cornerState(Axis::Y | Axis::Z | Axis::X))
    && (child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::Z) == cornerState(Axis::Z) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::Z) == cornerState(Axis::Z | Axis::Y) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::Z) == cornerState(Axis::Z | Axis::X) ||
      child(Axis::X | Axis::Y | Axis::Z)->cornerState(Axis::Z) ==
      cornerState(Axis::Z | Axis::Y | Axis::X));

  const bool center_safe =
    child(0)->cornerState(Axis::X | Axis::Y | Axis::Z) == cornerState(0) ||
    child(0)->cornerState(Axis::X | Axis::Y | Axis::Z) == cornerState(Axis::X) ||
    child(0)->cornerState(Axis::X | Axis::Y | Axis::Z) == cornerState(Axis::Y) ||
    child(0)->cornerState(Axis::X | Axis::Y | Axis::Z) == cornerState(Axis::X | Axis::Y) ||
    child(0)->cornerState(Axis::X | Axis::Y | Axis::Z) == cornerState(Axis::Z) ||
    child(0)->cornerState(Axis::X | Axis::Y | Axis::Z) == cornerState(Axis::Z | Axis::X) ||
    child(0)->cornerState(Axis::X | Axis::Y | Axis::Z) == cornerState(Axis::Z | Axis::Y) ||
    child(0)->cornerState(Axis::X | Axis::Y | Axis::Z) == cornerState(Axis::Z | Axis::X | Axis::Y);

  return edges_safe && faces_safe && center_safe;
}

template <>
bool XTree<3>::cornersAreManifold() const
{
  /* The code to generate the table is given below:
  def safe(index):
  f = [(index & (1 << i)) != 0 for i in range(8)]
  edges = [(0,1), (0,2), (2,3), (1,3),
  (4,5), (4,6), (6,7), (5,7),
  (0,4), (2,6), (1,5), (3,7)]
  def merge(a, b):
  merged = [(e[0] if e[0] != a else b,
  e[1] if e[1] != a else b) for e in edges]
  return [e for e in merged if e[0] != e[1]]
  while True:
  for e in edges:
  if f[e[0]] == f[e[1]]:
  edges = merge(e[0], e[1])
  break
  else:
  break
  s = set(map(lambda t: tuple(sorted(t)),edges))
  return len(s) <= 1
  out = ""
  for i,s in enumerate([safe(i) for i in range(256)]):
  if out == "": out += "{"
  else: out += ","
  if i and i % 32 == 0:
  out += '\n '
  if s: out += "1"
  else: out += "0"
  out += "}"
  print(out)
  */
  const static bool corner_table[] =
  { 1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,
    1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,1,1,1,1,1,0,1,0,1,0,0,1,1,0,0,0,1,
    1,0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,
    1,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1,1,0,1,
    1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1 };
  return corner_table[corner_mask];
}

template <>
const std::vector<std::pair<uint8_t, uint8_t>>& XTree<3>::edges() const
{
  static const std::vector<std::pair<uint8_t, uint8_t>> es =
  { { 0, Axis::X },{ 0, Axis::Y },{ 0, Axis::Z },
  { Axis::X, Axis::X | Axis::Y },{ Axis::X, Axis::X | Axis::Z },
  { Axis::Y, Axis::Y | Axis::X },{ Axis::Y, Axis::Y | Axis::Z },
  { Axis::X | Axis::Y, Axis::X | Axis::Y | Axis::Z },
  { Axis::Z, Axis::Z | Axis::X },{ Axis::Z, Axis::Z | Axis::Y },
  { Axis::Z | Axis::X, Axis::Z | Axis::X | Axis::Y },
  { Axis::Z | Axis::Y, Axis::Z | Axis::Y | Axis::X } };
  return es;
}

////////////////////////////////////////////////////////////////////////////////

// Explicit initialization of templates and templated methods.
template class XTree<2>;
template class XTree<3>;

////////////////////////////////////////////////////////////////////////////////

static std::vector<float> data;
void getData(std::vector<float> *p_data) { *p_data = data; }

static std::vector<Eigen::Vector3f> dbgPoints;
static std::vector<Eigen::Vector3f> dbgPointColors;
static std::vector<Eigen::Vector3f> dbgLines;
static std::vector<Eigen::Vector3f> dbgLineColors;

void dbgAddPoint(Eigen::Vector3f xyz, Eigen::Vector3f rgb)
{
  dbgPoints.push_back(xyz);
  dbgPointColors.push_back(rgb);
}

void dbgAddLine(Eigen::Vector3f xyz1, Eigen::Vector3f xyz2, Eigen::Vector3f rgb)
{
  dbgLines.push_back(xyz1);
  dbgLines.push_back(xyz2);
  dbgLineColors.push_back(rgb);
}

void dbgAddBox(Eigen::Vector3f p000, Eigen::Vector3f p111, Eigen::Vector3f rgb)
{
  Eigen::Vector3f p001(p000[0], p000[1], p111[2]), p010(p000[0], p111[1], p000[2]),
    p011(p000[0], p111[1], p111[2]), p100(p111[0], p000[1], p000[2]),
    p101(p111[0], p000[1], p111[2]), p110(p111[0], p111[1], p000[2]);
  dbgAddLine(p000, p001, rgb);
  dbgAddLine(p001, p011, rgb);
  dbgAddLine(p011, p010, rgb);
  dbgAddLine(p010, p000, rgb);
  dbgAddLine(p100, p101, rgb);
  dbgAddLine(p101, p111, rgb);
  dbgAddLine(p111, p110, rgb);
  dbgAddLine(p110, p100, rgb);
  dbgAddLine(p000, p100, rgb);
  dbgAddLine(p001, p101, rgb);
  dbgAddLine(p011, p111, rgb);
  dbgAddLine(p010, p110, rgb);
}

void dbgGetPointsAndLines(std::vector<Eigen::Vector3f> &points, std::vector<Eigen::Vector3f> &pointColors,
  std::vector<Eigen::Vector3f> &lines, std::vector<Eigen::Vector3f> &lineColors)
{
  points = dbgPoints;
  pointColors = dbgPointColors;
  lines = dbgLines;
  lineColors = dbgLineColors;
}

void dbgClear()
{
  dbgPoints.resize(0);
  dbgPointColors.resize(0);
  dbgLines.resize(0);
  dbgLineColors.resize(0);
}

}   // namespace Kernel
