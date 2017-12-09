#include "ao/eval/eval_point.hpp"

namespace Kernel {

PointEvaluator::PointEvaluator(std::shared_ptr<Tape> t)
    : PointEvaluator(t, std::map<Tree::Id, float>())
{
    // Nothing to do here
}

PointEvaluator::PointEvaluator(
        std::shared_ptr<Tape> t, const std::map<Tree::Id, float>& vars)
    : BaseEvaluator(t, vars), f(tape->num_clauses + 1, 1)
{
    // Unpack variables into result array
    for (auto& v : t->vars.right)
    {
        auto var = vars.find(v.first);
        f(v.second) = (var != vars.end()) ? var->second : 0;
    }

    // Unpack constants into result array
    for (auto& c : tape->constants)
    {
        f(c.first) = c.second;
    }
}

float PointEvaluator::eval(const Eigen::Vector3f& pt)
{
    if (tape->XOpc)
      f(tape->XOpc) = pt.x();
    if (tape->YOpc)
      f(tape->YOpc) = pt.y();
    if (tape->ZOpc)
      f(tape->ZOpc) = pt.z();        
    for (auto prim : tape->primitives) {
      f(prim.first) = prim.second->getValue(pt);
    }

    return f(tape->rwalk(*this));
}

float PointEvaluator::evalAndPush(const Eigen::Vector3f& pt)
{
    auto out = eval(pt);
    tape->push([&](Opcode::Opcode op, Clause::Id /* id */,
                  Clause::Id a, Clause::Id b)
    {
        // For min and max operations, we may only need to keep one branch
        // active if it is decisively above or below the other branch.
        if (op == Opcode::MAX)
        {
            if (f(a) > f(b))
            {
                return Tape::KEEP_A;
            }
            else if (f(b) > f(a))
            {
                return Tape::KEEP_B;
            }
        }
        else if (op == Opcode::MIN)
        {
            if (f(a) > f(b))
            {
                return Tape::KEEP_B;
            }
            else if (f(b) > f(a))
            {
                return Tape::KEEP_A;
            }
        }
        return Tape::KEEP_BOTH;
    }, Tape::SPECIALIZED);
    return out;
}

float PointEvaluator::baseEval(const Eigen::Vector3f& pt)
{
    return tape->baseEval<PointEvaluator, float>(*this, pt);
}

////////////////////////////////////////////////////////////////////////////////

bool PointEvaluator::setVar(Tree::Id var, float value)
{
    auto v = tape->vars.right.find(var);
    if (v != tape->vars.right.end())
    {
        bool changed = f(v->second) != value;
        f.row(v->second) = value;
        return changed;
    }
    else
    {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////

void PointEvaluator::operator()(Opcode::Opcode op, Clause::Id id,
                                Clause::Id a, Clause::Id b)
{
#define out f(id)
#define a f(a)
#define b f(b)
    switch (op)
    {
        case Opcode::ADD:
            out = a + b;
            break;
        case Opcode::MUL:
            out = a * b;
            break;
        case Opcode::MIN:
            out = fmin(a, b);
            break;
        case Opcode::MAX:
            out = fmax(a, b);
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
            out = pow(a, b);
            break;
        case Opcode::NTH_ROOT:
            out = pow(a, 1.0f/b);
            break;
        case Opcode::MOD:
            out = std::fmod(a, b);
            while (out < 0)
            {
                out += b;
            }
            break;
        case Opcode::NANFILL:
            out = std::isnan(a) ? b : a;
            break;

        case Opcode::USEINTERVAL:
          out = a;
          break;

        case Opcode::CLEANUNION:
          out = a + b - sqrt(a * a + b * b);
          break;

        case Opcode::CLEANINTERSECT:
          out = a + b + sqrt(a * a + b * b);
          break;

        case Opcode::SQUARE:
            out = a * a;
            break;
        case Opcode::SQRT:
            out = sqrt(a);
            break;
        case Opcode::NEG:
            out = -a;
            break;
        case Opcode::SIN:
            out = sin(a);
            break;
        case Opcode::COS:
            out = cos(a);
            break;
        case Opcode::TAN:
            out = tan(a);
            break;
        case Opcode::ASIN:
            out = asin(a);
            break;
        case Opcode::ACOS:
            out = acos(a);
            break;
        case Opcode::ATAN:
            out = atan(a);
            break;
        case Opcode::EXP:
            out = exp(a);
            break;
        case Opcode::ABS:
            out = fabs(a);
            break;
        case Opcode::RECIP:
            out = 1 / a;
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


