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
#include "libfive/eval/eval_deriv_array.hpp"

namespace Kernel {

DerivArrayEvaluator::DerivArrayEvaluator(std::shared_ptr<Tape> t)
    : DerivArrayEvaluator(t, std::map<Tree::Id, float>())
{
    // Nothing to do here
}

DerivArrayEvaluator::DerivArrayEvaluator(
        std::shared_ptr<Tape> t, const std::map<Tree::Id, float>& vars)
    : ArrayEvaluator(t, vars), d(tape->num_clauses + 1, 1)
{
    // Initialize all derivatives to zero
    for (unsigned i=0; i < d.rows(); ++i)
    {
        d(i) = 0;
    }

    // Load immutable derivatives for X, Y, Z
    d(tape->X).row(0) = 1;
    d(tape->Y).row(1) = 1;
    d(tape->Z).row(2) = 1;
}

Eigen::Vector4f DerivArrayEvaluator::deriv(const Eigen::Vector3f& pt)
{
    set(pt, 0);
    return derivs(1).col(0);
}

Eigen::Block<decltype(DerivArrayEvaluator::out), 4, Eigen::Dynamic>
DerivArrayEvaluator::derivs(size_t count)
{
    // Perform value evaluation, copying results into the 4th row of out
    out.row(3).head(count) = values(count);

    // Perform derivative evaluation, copying results into the out array
    out.topLeftCorner(3, count) = d(tape->rwalk(*this)).leftCols(count);

    // Return a block of valid results from the out array
    return out.block<4, Eigen::Dynamic>(0, 0, 4, count);
}

void DerivArrayEvaluator::operator()(Opcode::Opcode op, Clause::Id id,
                                     Clause::Id a, Clause::Id b)
{
#define ov f.row(id).head(count)
#define od d(id).leftCols(count)

#define av f.row(a).head(count)
#define ad d(a).leftCols(count)

#define bv f.row(b).head(count)
#define bd d(b).leftCols(count)

    switch (op) {
        case Opcode::ADD:
            od = ad + bd;
            break;
        case Opcode::MUL:
            // Product rule
            od = bd.rowwise()*av + ad.rowwise()*bv;
            break;
        case Opcode::MIN:
            for (unsigned i=0; i < od.rows(); ++i)
                od.row(i) = (av < bv).select(ad.row(i), bd.row(i));
            break;
        case Opcode::MAX:
            for (unsigned i=0; i < od.rows(); ++i)
                od.row(i) = (av < bv).select(bd.row(i), ad.row(i));
            break;
        case Opcode::SUB:
            od = ad - bd;
            break;
        case Opcode::DIV:
          for (auto i = 0; i < count; ++i) {
            d(id).col(i) = d(a).col(i) * f(b, i) - d(b).col(i) * f(a, i) / (f(b, i) * f(b, i));
          }
          /*od = (ad.rowwise()*bv - bd.rowwise()*av).rowwise() /
          bv.pow(2);*/
            break;
        case Opcode::ATAN2:
            od = (ad.rowwise()*bv - bd.rowwise()*av).rowwise() /
                 (av.pow(2) + bv.pow(2));
            break;
        case Opcode::POW:
            // The full form of the derivative is
            // od = m * (bv * ad + av * log(av) * bd))
            // However, log(av) is often NaN and bd is always zero,
            // (since it must be CONST), so we skip that part.
            od = ad.rowwise() * (bv * av.pow(bv - 1));
            break;

        case Opcode::NTH_ROOT:
            for (unsigned i=0; i < od.cols(); ++i)
                od.col(i) = (ad.col(i) == 0)
                    .select(0, ad.col(i) * (pow(av(i), 1.0f / bv(i) - 1) / bv(i)));
            break;
        case Opcode::MOD:
            od = ad;
            break;
        case Opcode::NANFILL:
            for (unsigned i=0; i < od.rows(); ++i)
                od.row(i) = av.isNaN().select(bd.row(i), ad.row(i));
            break;

        case Opcode::SQUARE:
            od = ad.rowwise() * av * 2;
            break;
        case Opcode::SQRT:
            for (unsigned i=0; i < od.rows(); ++i)
                od.row(i) = (av < 0).select(
                    Eigen::Array<float, 1, Eigen::Dynamic>::Zero(1, count),
                    ad.row(i) / (2 * ov));
            break;
        case Opcode::NEG:
            od = -ad;
            break;
        case Opcode::SIN:
            od = ad.rowwise() * cos(av);
            break;
        case Opcode::COS:
            od = ad.rowwise() * -sin(av);
            break;
        case Opcode::TAN:
            od = ad.rowwise() * pow(1/cos(av), 2);
            break;
        case Opcode::ASIN:
            od = ad.rowwise() / sqrt(1 - pow(av, 2));
            break;
        case Opcode::ACOS:
            od = ad.rowwise() / -sqrt(1 - pow(av, 2));
            break;
        case Opcode::ATAN:
            od = ad.rowwise() / (pow(av, 2) + 1);
            break;
        case Opcode::LOG:
            od = ad.rowwise() / av;
            break;
        case Opcode::EXP:
            od = ad.rowwise() * exp(av);
            break;
        case Opcode::ABS:
            for (unsigned i=0; i < od.rows(); ++i)
                od.row(i) = (av > 0).select(ad.row(i), -ad.row(i));
            break;
        case Opcode::RECIP:
            od = ad.rowwise() / -av.pow(2);
            break;

        case Opcode::CONST_VAR:
            od = ad;
            break;

        case Opcode::INVALID:
        case Opcode::CONST:
        case Opcode::VAR_X:
        case Opcode::VAR_Y:
        case Opcode::VAR_Z:
        case Opcode::VAR:
        case Opcode::LAST_OP: assert(false);
    }
#undef ov
#undef od

#undef av
#undef ad

#undef bv
#undef bd

}

}   // namespace Kernel
