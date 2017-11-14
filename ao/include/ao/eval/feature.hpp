#pragma once

#include <list>
#include <map>
#include <set>

#include <Eigen/Eigen>

#include "ao/eval/clause.hpp"

namespace Kernel
{

/*
 *  A feature corresponds to a set of choices (e.g. whether to take the minimum or 
 *  maximum where they are equal but their derivatives are not).  Each feature also
 *  stores a collection of epsilons (normalized vectors); the feature is only
 *  present in the vicinity of the point in directions that have no more than a 90
 *  degree angle from all epsilons.
 */  
class Feature
{
public:
    struct Choice
    {
        const Clause::Id id;
        const int choice;
    };

    struct PrimitiveChoice
    {
      const Clause::Id id;
      const Eigen::Vector3f choice;
    };

    /*
     *  Checks to see whether a particular epsilon is compatible with
     *  all of the other epsilons in the system.
     *  This is a slow (worst-case O(n^3)) operation, but it should be called
     *  rarely and so doesn't need to be optimized yet.
     */
    bool isCompatible(Eigen::Vector3d e) const;

    /*
     *  If incompatible, does nothing and returns false
     *  Otherwise, pushes to the front of the choice list and returns true
     */
    bool push(Eigen::Vector3d e, Choice c={0, 0});
    bool push(Eigen::Vector3d e, PrimitiveChoice c);

    /*
     *  Accessor method for the choice lists
     */
    const std::set<Choice>& getChoices() const { return choices; }
    const std::set<PrimitiveChoice>& getPrimitiveChoices() const {return primitiveChoices; }

    /*
     *  Top-level derivative (set manually)
     */
    Eigen::Vector3d deriv;

    /*
     *  Inserts a choice without any checking
     */
    void pushRaw(Choice c, Eigen::Vector3d v);

    /*
     *  Inserts a choice without an associated direction
     *  This is useful to collapse cases like min(a, a)
     */
    void pushChoice(Choice c);

    /*
     *  Returns the epsilon associated with a particular choice
     */
    Eigen::Vector3d getEpsilon(Clause::Id i) const { return _epsilons.at(i); }

    /*
     *  Checks to see whether the given clause has an epsilon
     */
    bool hasEpsilon(Clause::Id i) const
        { return _epsilons.find(i) != _epsilons.end(); }

protected:
    /*
     *  Versions of isCompatible and push when e is known to be normalized
     */
    bool isCompatibleNorm(Eigen::Vector3d e) const;
    bool pushNorm(Eigen::Vector3d e, Choice choice);
    bool pushNorm(Eigen::Vector3d e, PrimitiveChoice choice);

    typedef enum { NOT_PLANAR, PLANAR_FAIL, PLANAR_SUCCESS } PlanarResult;
    PlanarResult checkPlanar(Eigen::Vector3d v) const;

    /*  Per-clause decisions  */
    std::set<Choice> choices;

    /*  Per-primitive decisions  */
    std::set<PrimitiveChoice> primitiveChoices;

    /*  Deduplicated list of epsilons  */
    std::list<Eigen::Vector3d> epsilons;

    /*  Per-clause epsilons  */
    std::map<Clause::Id, Eigen::Vector3d> _epsilons;
};

/*  Defining operator< lets us store Choices in std::set, etc */
bool operator<(const Feature::Choice& a, const Feature::Choice& b);
bool operator<(const Feature::PrimitiveChoice& a, const Feature::PrimitiveChoice& b);

}   // namespace Kernel
