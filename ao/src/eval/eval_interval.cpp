#include "ao/eval/eval_interval.hpp"

namespace Kernel {

IntervalEvaluator::IntervalEvaluator(std::shared_ptr<Tape> t)
    : IntervalEvaluator(t, std::map<Tree::Id, float>())
{
    // Nothing to do here
}

IntervalEvaluator::IntervalEvaluator(
        std::shared_ptr<Tape> t, const std::map<Tree::Id, float>& vars)
    : BaseEvaluator(t, vars)
{
    i.resize(tape->num_clauses + 1);

    // Unpack variables into result array
    for (auto& v : t->vars.right)
    {
        auto var = vars.find(v.first);
        i[v.second] = (var != vars.end()) ? var->second : 0;
    }

    // Unpack constants into result array
    for (auto& c : tape->constants)
    {
        i[c.first] = c.second;
    }
}

Interval::I IntervalEvaluator::eval(const Eigen::Vector3f& lower,
                                    const Eigen::Vector3f& upper)
{
    i[tape->XOpc] = {lower.x(), upper.x()};
    i[tape->YOpc] = {lower.y(), upper.y()};
    i[tape->ZOpc] = {lower.z(), upper.z()};
    Region<3> region(lower.template cast<double>(), upper.template cast<double>());
    for (auto prim : tape->primitives) {
      i[prim.first] = prim.second->getRange(region);
    }

    return i[tape->rwalk(*this)];
}

Interval::I IntervalEvaluator::evalAndPush(const Eigen::Vector3f& lower,
                                           const Eigen::Vector3f& upper)
{
    auto out = eval(lower, upper);

    tape->push([&](Opcode::Opcode op, Clause::Id /* id */,
                  Clause::Id a, Clause::Id b)
    {
        // For min and max operations, we may only need to keep one branch
        // active if it is decisively above or below the other branch.
        if (op == Opcode::MAX)
        {
            if (i[a].lower() > i[b].upper())
            {
                return Tape::KEEP_A;
            }
            else if (i[b].lower() > i[a].upper())
            {
                return Tape::KEEP_B;
            }
        }
        else if (op == Opcode::MIN)
        {
            if (i[a].lower() > i[b].upper())
            {
                return Tape::KEEP_B;
            }
            else if (i[b].lower() > i[a].upper())
            {
                return Tape::KEEP_A;
            }
        }
        return Tape::KEEP_BOTH;
    },
        Tape::INTERVAL,
        {lower.template cast<double>(), upper.template cast<double>() });
    return out;
}

////////////////////////////////////////////////////////////////////////////////

bool IntervalEvaluator::setVar(Tree::Id var, float value)
{
    auto v = tape->vars.right.find(var);
    if (v != tape->vars.right.end())
    {
        bool changed = i[v->second] != value;
        i[v->second] = value;
        return changed;
    }
    else
    {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////

void IntervalEvaluator::operator()(Opcode::Opcode op, Clause::Id id,
                                   Clause::Id a, Clause::Id b)
{
#define out i[id]
#define a i[a]
#define b i[b]
    switch (op) {
        case Opcode::ADD:
            out = a + b;
            break;
        case Opcode::MUL:
            out = a * b;
            break;
        case Opcode::MIN:
            out = boost::numeric::min(a, b);
            break;
        case Opcode::MAX:
            out = boost::numeric::max(a, b);
            break;
        case Opcode::SUB:
            out = a - b;
            break;
        case Opcode::DIV:
            out = a / b;
            break;
        case Opcode::ATAN2:
            out = atan2(a, b);
            break;
        case Opcode::POW:
            out = boost::numeric::pow(a, b.lower());
            break;
        case Opcode::NTH_ROOT:
            out = boost::numeric::nth_root(a, b.lower());
            break;
        case Opcode::MOD:
            out = Interval::I(0.0f, b.upper()); // YOLO
            break;
        case Opcode::NANFILL:
            out = (std::isnan(a.lower()) || std::isnan(a.upper())) ? b : a;
            break;

        case Opcode::USEINTERVAL:
            out = b;
            break;

        case Opcode::CLEANUNION:
            out = Interval::I(a.lower() + b.lower() - sqrt(a.lower() * a.lower() + b.lower() * b.lower()),
              a.upper() + b.upper() - sqrt(a.upper() * a.upper() + b.upper() * b.upper())); //Works because it's an increasing function.
            break;

        case Opcode::CLEANINTERSECT:
            out = Interval::I(a.lower() + b.lower() + sqrt(a.lower() * a.lower() + b.lower() * b.lower()),
              a.upper() + b.upper() + sqrt(a.upper() * a.upper() + b.upper() * b.upper()));
            break;

        case Opcode::SQUARE:
            out = boost::numeric::square(a);
            break;
        case Opcode::SQRT:
            out = boost::numeric::sqrt(a);
            break;
        case Opcode::NEG:
            out = -a;
            break;
        case Opcode::SIN:
            out = boost::numeric::sin(a);
            break;
        case Opcode::COS:
            out = boost::numeric::cos(a);
            break;
        case Opcode::TAN:
            out = boost::numeric::tan(a);
            break;
        case Opcode::ASIN:
            out = boost::numeric::asin(a);
            break;
        case Opcode::ACOS:
            out = boost::numeric::acos(a);
            break;
        case Opcode::ATAN:
            // If the interval has an infinite bound, then return the largest
            // possible output interval (of +/- pi/2).  This rescues us from
            // situations where we do atan(y / x) and the behavior of the
            // interval changes if you're approaching x = 0 from below versus
            // from above.
            out = (std::isinf(a.lower()) || std::isinf(a.upper()))
                ? Interval::I(-M_PI/2, M_PI/2)
                : boost::numeric::atan(a);
            break;
        case Opcode::EXP:
            out = boost::numeric::exp(a);
            break;
        case Opcode::ABS:
            out = boost::numeric::abs(a);
            break;
        case Opcode::RECIP:
            out = Interval::I(1,1) / a;
            break;

        case Opcode::CONST_VAR:
            out = a;
            break;

        case Opcode::INVALID:
        case Opcode::CONST:
        case Opcode::VAR_X:
        case Opcode::VAR_Y:
        case Opcode::VAR_Z:
        case Opcode::VAR:
        case Opcode::PRIMITIVE:
        case Opcode::LAST_OP: assert(false);
    }
#undef out
#undef a
#undef b
}

}   // namespace Kernel
