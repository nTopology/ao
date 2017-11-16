#include "ao/eval/eval_feature.hpp"

namespace Kernel {

FeatureEvaluator::FeatureEvaluator(std::shared_ptr<Tape> t)
    : FeatureEvaluator(t, std::map<Tree::Id, float>())
{
    // Nothing to do here
}

FeatureEvaluator::FeatureEvaluator(
        std::shared_ptr<Tape> t, const std::map<Tree::Id, float>& vars)
    : DerivEvaluator(t, vars), dPrimAll(tape->num_clauses + 1)
{
    // Nothing to do here
}

Eigen::Vector4f FeatureEvaluator::deriv(const Eigen::Vector3f& pt)
{
  //Load gradients of primitives; only use the first gradient of each here 
  //(following precendent for min and max evaluation). 
  for (auto prim : tape->primitives) {
    dPrimAll(prim.first) = prim.second->getGradients(pt); //Other than this line, it's the same as the DerivEvaluator version.
    d.col(prim.first) = *(dPrimAll(prim.first).begin());
  }
  // Perform value evaluation, saving results
  auto w = eval(pt);
  auto xyz = d.col(tape->rwalk(*this));
  Eigen::Vector4f out;
  out << xyz, w;
  return out;
}

Eigen::Vector4f FeatureEvaluator::deriv(const Eigen::Vector3f& pt, const Feature& feature)
{
  if (feature.getPrimitiveChoices().empty())
    return deriv(pt);
  //Load gradients of ambiguous primitives (non-ambiguous ones are assumed to have already been loaded).
  for (auto prim : feature.getPrimitiveChoices()) {
    d.col(prim.id) = prim.choice;
  }
  // Perform derivative evaluation, saving results (value evaluation is assumed to have already been done.)
  auto w = eval(pt);
  auto xyz = d.col(tape->rwalk(*this));

  Eigen::Vector4f out;
  out << xyz, w;
  return out;
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
        if ((op == Opcode::MAX || op == Opcode::MIN) &&
            (f(a, 0) == f(b, 0) || a == b))
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
            [&](Opcode::Opcode op, Clause::Id id, Clause::Id a, Clause::Id b)
            {
                ambig |= (op == Opcode::MIN || op == Opcode::MAX) &&
                         (f(a) == f(b));
                ambig |= (op == Opcode::PRIMITIVE && dPrimAll(id).size() > 1);
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

        // Evaluate the derivatives at this point.
        // The value will be the same throughout, but derivatives may change
        // depending on which feature we're in.
        // The first call has a blank feature, so populates non-ambiguous primitives
        // and gets and stores all gradients for all primitives.
        const Eigen::Vector3f ds = deriv(p, feature).template head<3>();

        bool ambiguous = false;
        tape->rwalk(
            [&](Opcode::Opcode op, Clause::Id id, Clause::Id a, Clause::Id b)
            {
                if (op == Opcode::PRIMITIVE && dPrimAll(id).size() > 1) 
                {
                    const auto& choices = feature.getChoices();
                    auto itr = choices.begin();
                    while (itr != choices.end() && itr->id != id) { itr++; }
                    if (itr == choices.end()) 
                    {
                      for (auto choice : dPrimAll(id)) 
                      {
                        auto fNew = f_;
                        for (auto choice2 : dPrimAll(id)) {
                          if (choice != choice2) {
                            Primitive::prioritizationType prioritization = tape->primitives[id]->getPrioritizationType(p);
                            Eigen::Vector3f epsilon;
                            if (prioritization == Primitive::e_PRIMITIVE_MINIMUM)
                              epsilon = choice2 - choice;
                            else if (prioritization == Primitive::e_PRIMITIVE_MAXIMUM)
                              epsilon = choice - choice2;
                            if (!fNew.push(epsilon.template cast<double>(), { id, choice }))
                              goto continueOuterLoop;
                          }
                        }
                        ambiguous = true;
                        todo.push_back(fNew);
                        continueOuterLoop:
                        ambiguous = ambiguous; //Because labels can't be at the end of blocks
                      }
                    }
                }
                else if ((op == Opcode::MIN || op == Opcode::MAX))
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
