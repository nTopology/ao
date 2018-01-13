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
#pragma once

#include <cstdlib>
#include <string>

namespace Kernel {

namespace Opcode
{
// To create a new opcode, add it to the relevant section with the
// next-highest value and increment LAST_OP by one.  This is necessary
// to avoid changing the meanings of opcodes in previously saved files.
#define OPCODES \
    OPCODE(INVALID, 0)      \
                            \
    OPCODE(CONST, 1)        \
    OPCODE(VAR_X, 2)        \
    OPCODE(VAR_Y, 3)        \
    OPCODE(VAR_Z, 4)        \
    OPCODE(VAR, 5)          \
    OPCODE(CONST_VAR, 6)    \
                            \
    OPCODE(SQUARE, 7)       \
    OPCODE(SQRT, 8)         \
    OPCODE(NEG, 9)          \
    OPCODE(SIN, 10)         \
    OPCODE(COS, 11)         \
    OPCODE(TAN, 12)         \
    OPCODE(ASIN, 13)        \
    OPCODE(ACOS, 14)        \
    OPCODE(ATAN, 15)        \
    OPCODE(EXP, 16)         \
    OPCODE(ABS, 28)         \
    OPCODE(LOG, 30)         \
    OPCODE(RECIP, 29)       \
                            \
    OPCODE(ADD, 17)         \
    OPCODE(MUL, 18)         \
    OPCODE(MIN, 19)         \
    OPCODE(MAX, 20)         \
    OPCODE(SUB, 21)         \
    OPCODE(DIV, 22)         \
    OPCODE(ATAN2, 23)       \
    OPCODE(POW, 24)         \
    OPCODE(NTH_ROOT, 25)    \
    OPCODE(MOD, 26)         \
    OPCODE(NANFILL, 27)

enum Opcode {
#define OPCODE(s, i) s=i,
    OPCODES
#undef OPCODE
    LAST_OP=31,
};

size_t args(Opcode op);

/*
 *  Converts to the bare enum string (e.g. ATAN2)
 */
std::string toString(Opcode op);

/*
 *  Converts to a operator string (+, -, etc) or the function name
 *  (atan, cos, etc)
 */
std::string toOpString(Opcode op);

/*
 *  Returns a Scheme symbol-style string, e.g. lower-case
 *  with underscores replaced by dashes.
 */
std::string toScmString(Opcode op);

/*
 *  Converts from a Scheme symbol-style string to an enum value,
 *  return INVALID if there's no match.
 */
Opcode fromScmString(std::string s);

/*
 *  Returns true if the opcode is commutative (+, *, etc)
 */
bool isCommutative(Opcode op);

}


}   // namespace Kernel
