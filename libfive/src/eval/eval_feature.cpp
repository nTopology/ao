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
#include "libfive/eval/eval_feature.hpp"

namespace Kernel {

FeatureEvaluator::FeatureEvaluator(std::shared_ptr<Tape> t)
    : FeatureEvaluator(t, std::map<Tree::Id, float>())
{
    // Nothing to do here
}

FeatureEvaluator::FeatureEvaluator(
        std::shared_ptr<Tape> t, const std::map<Tree::Id, float>& vars)
    : DerivEvaluator(t, vars)
{
    // Nothing to do here
}

Feature FeatureEvaluator::push(const Feature& feature)
{
    Feature out;
    out.deriv = feature.deriv;

    const auto& choices = feature.getChoices();
    auto itr = choices.begin();

    tape->push([&](Opcode::Opcode op, Clause::Id id, Clause::Id a, Clause::Id b)
    {
        // First, check whether this is an ambiguous operation
        // If it is, then there may be a choice for it in the feature
        if (op == Opcode::MAX || op == Opcode::MIN)
        {
            if (f(a, 0) == f(b, 0) || a == b)
            {
                // Walk the iterator forwards until we find a match by id
                // or hit the end of the feature
                while (itr != choices.end() && itr->id < id) { itr++; }

                // Push either the choice + epsilon or the bare choice to the
                // output feature, effectively pruning it to contain only the
                // choices that are actually significant in this subtree.
                if (itr != choices.end() && itr->id == id)
                {
                    if (feature.hasEpsilon(id))
                    {
                        out.pushRaw(*itr, feature.getEpsilon(id));
                    }
                    else
                    {
                        out.pushChoice(*itr);
                    }

                    return (itr->choice == 0) ? Tape::KEEP_A : Tape::KEEP_B;
                }
            }
            return Tape::KEEP_BOTH;
        }
        else
        {
            return Tape::KEEP_ALWAYS;
        }
    }, Tape::FEATURE);

    return out;
}

bool FeatureEvaluator::isInside(const Eigen::Vector3f& p)
{
    auto ds = deriv(p);

    // Unambiguous cases
    if (ds.w() < 0)
    {
        return true;
    }
    else if (ds.w() > 0)
    {
        return false;
    }

    // Special case to save time on non-ambiguous features: we can get both
    // positive and negative values out if there's a non-zero gradient
    // (same as single-feature case below).
    {
        bool ambig = false;
        tape->walk(
            [&](Opcode::Opcode op, Clause::Id /* id */, Clause::Id a, Clause::Id b)
            {
                ambig |= (op == Opcode::MIN || op == Opcode::MAX) &&
                         (f(a) == f(b));
            }, ambig);

        if (!ambig)
        {
            return (ds.col(0).template head<3>().array() != 0).any();
        }
    }

    // Otherwise, we need to handle the zero-crossing case!

    // First, we extract all of the features
    auto fs = featuresAt(p);

    // If there's only a single feature, we can get both positive and negative
    // values out if it's got a non-zero gradient
    if (fs.size() == 1)
    {
        return fs.front().deriv.norm() > 0;
    }

    // Otherwise, check each feature
    // The only case where we're outside the model is if all features
    // and their normals are all positive (i.e. for every epsilon that
    // we move from (x,y,z), epsilon . deriv > 0)
    bool pos = false;
    bool neg = false;
    for (auto& f : fs)
    {
        pos |= f.isCompatible(f.deriv);
        neg |= f.isCompatible(-f.deriv);
    }
    return !(pos && !neg);

}

std::list<Feature> FeatureEvaluator::featuresAt(const Eigen::Vector3f& p)
{
    // The initial feature doesn't know any ambiguities
    Feature feature;
    std::list<Feature> todo = {feature};
    std::list<Feature> done;
    std::set<std::set<Feature::Choice>> seen;

    // Load the location into the first results slot and evaluate
    evalAndPush(p);

    while (todo.size())
    {
        // Take the most recent feature and scan for ambiguous min/max nodes
        // (from the bottom up).  If we find such an ambiguous node, then push
        // both versions to the feature (if compatible) and re-insert the
        // augmented feature in the todo list; otherwise, move the feature
        // to the done list.
        auto feature = todo.front();
        todo.pop_front();

        // Then, push into this feature
        // (storing a minimized version of the feature)
        auto f_ = push(feature);

        // Run a single evaluation of the value + derivatives
        // The value will be the same, but derivatives may change
        // depending on which feature we've pushed ourselves into
        const Eigen::Vector3f ds = deriv(p).template head<3>();

        bool ambiguous = false;
        tape->rwalk(
            [&](Opcode::Opcode op, Clause::Id id, Clause::Id a, Clause::Id b)
            {
                if ((op == Opcode::MIN || op == Opcode::MAX))
                {
                    // If we've ended up with a non-selection, then collapse
                    // it to a single choice
                    if (a == b)
                    {
                        auto fa = f_;
                        fa.pushChoice({id, 0});
                        todo.push_back(fa);
                        ambiguous = true;
                    }
                    // Check for ambiguity here
                    else if (f(a, 0) == f(b, 0))
                    {
                        // Check both branches of the ambiguity
                        const Eigen::Vector3d rhs(
                                d.col(b).template cast<double>());
                        const Eigen::Vector3d lhs(
                                d.col(a).template cast<double>());
                        const auto epsilon = (op == Opcode::MIN) ? (rhs - lhs)
                                                                      : (lhs - rhs);

                        auto fa = f_;
                        if (fa.push(epsilon, {id, 0}))
                        {
                            ambiguous = true;
                            todo.push_back(fa);
                        }

                        auto fb = f_;
                        if (fb.push(-epsilon, {id, 1}))
                        {
                            ambiguous = true;
                            todo.push_back(fb);
                        }
                    }
                }
            }, ambiguous);

        if (!ambiguous)
        {
            f_.deriv = ds.col(0).template head<3>().template cast<double>();
            if (seen.find(f_.getChoices()) == seen.end())
            {
                seen.insert(f_.getChoices());
                done.push_back(f_);
            }
        }
        pop(); // push(Feature)
    }
    pop(); // specialization

    assert(done.size() > 0);
    return done;

}

}   // namespace Kernel
