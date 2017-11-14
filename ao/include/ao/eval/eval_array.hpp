#pragma once

#include <Eigen/Eigen>

#include "ao/eval/base.hpp"

namespace Kernel {

class ArrayEvaluator : public BaseEvaluator
{
public:
    ArrayEvaluator(std::shared_ptr<Tape> t);
    ArrayEvaluator(std::shared_ptr<Tape> t,
                   const std::map<Tree::Id, float>& vars);

    /*
     *  Single-point evaluation
     */
    float eval(const Eigen::Vector3f& pt);
    float evalAndPush(const Eigen::Vector3f& pt);

    /*
     *  Stores the given value in the result arrays
     *  (inlined for efficiency)
     */
    void set(const Eigen::Vector3f& p, size_t index)
    {
        if (tape->XOpc != 0)
          f(tape->XOpc, index) = p.x();
        if (tape->YOpc != 0) 
          f(tape->YOpc, index) = p.y();
        if (tape->ZOpc != 0)
          f(tape->ZOpc, index) = p.z();
        points(index) = p;
        for (auto prim : tape->primitives) {
          f(prim.first, index) = prim.second->getValue(p);
        }
    }

    /*  This is the number of samples that we can process in one pass */
    static constexpr size_t N=256;

protected:
    /*  Stored in values() and used in operator() to decide how much of the
     *  array we're addressing at once  */
    size_t count;

    /*  f(clause, index) is a specific data point */
    Eigen::Array<float, Eigen::Dynamic, N, Eigen::RowMajor> f;

    //Stores the points evaluated at each index.

    Eigen::Array<Eigen::Vector3f, 1, N> points;

    /*  ambig(index) returns whether a particular slot is ambiguous.*/
    Eigen::Array<bool, 1, N> ambig;

    /*
     *  Per-clause evaluation, used in tape walking
     */
    void operator()(Opcode::Opcode op, Clause::Id id,
                    Clause::Id a, Clause::Id b);
public:
    /*
     *  Multi-point evaluation (values must be stored with set)
     */
    Eigen::Block<decltype(f), 1, Eigen::Dynamic> values(size_t count);

    /*
     *  Changes a variable's value
     *
     *  If the variable isn't present in the tree, does nothing
     *  Returns true if the variable's value changes
     */
    bool setVar(Tree::Id var, float value);

    /*
     *  Returns a list of ambiguous items from indices 0 to i
     *
     *  This call performs O(i) work to set up the ambig array
     */
    Eigen::Block<decltype(ambig), 1, Eigen::Dynamic> getAmbiguous(size_t i);

    /*  Make an aligned new operator, as this class has Eigen structs
     *  inside of it (which are aligned for SSE) */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class Tape; // for rwalk<ArrayEvaluator>
};

}   // namespace Kernel

