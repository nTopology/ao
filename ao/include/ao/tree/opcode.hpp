#pragma once

#include <cstdlib>
#include <string>

namespace Kernel {

namespace Opcode
{
// To create a new opcode, add it to the relevant section with the
// next-highest value and increment LAST_OP by one.  This is necessary
// to avoid changing the meanings of opcodes in previously saved files.
// As a result, opcodes are not all in order.

//The primitives system makes VAR_X, VAR_Y, and VAR_Z deprecated
//unless the ao i/o methods are needed; PRIMITIVE with an appropriate
//reference should be used instead.

//USEINTERVAL uses the values and derivatives of the first argument,
//but the interval of the second; it can therefore allow for more
//efficient/tight interval computation of particular functions.
//CLEANUNIONINTERVAL and CLEANINTERSECTINTERVAL provide the clean union 
//and clean intersection (x+y-sqrt(x^2+y^2) and x+y+sqrt(x^2+y^2)) 
//operators.
#define OPCODES \
    OPCODE(INVALID, 0)          \
                                \
    OPCODE(CONST, 1)            \
    OPCODE(VAR_X, 2)            \
    OPCODE(VAR_Y, 3)            \
    OPCODE(VAR_Z, 4)            \
    OPCODE(VAR, 5)              \
    OPCODE(CONST_VAR, 6)        \
    OPCODE(PRIMITIVE, 30)       \
                                \
    OPCODE(USEINTERVAL, 31)     \
    OPCODE(CLEANUNION, 32)      \
    OPCODE(CLEANINTERSECT, 33)  \
                                \
                                \
                                \
    OPCODE(SQUARE, 7)           \
    OPCODE(SQRT, 8)             \
    OPCODE(NEG, 9)              \
    OPCODE(SIN, 10)             \
    OPCODE(COS, 11)             \
    OPCODE(TAN, 12)             \
    OPCODE(ASIN, 13)            \
    OPCODE(ACOS, 14)            \
    OPCODE(ATAN, 15)            \
    OPCODE(EXP, 16)             \
    OPCODE(ABS, 28)             \
    OPCODE(RECIP, 29)           \
                                \
    OPCODE(ADD, 17)             \
    OPCODE(MUL, 18)             \
    OPCODE(MIN, 19)             \
    OPCODE(MAX, 20)             \
    OPCODE(SUB, 21)             \
    OPCODE(DIV, 22)             \
    OPCODE(ATAN2, 23)           \
    OPCODE(POW, 24)             \
    OPCODE(NTH_ROOT, 25)        \
    OPCODE(MOD, 26)             \
    OPCODE(NANFILL, 27)     
  

enum Opcode {
#define OPCODE(s, i) s=i,
    OPCODES
#undef OPCODE
    LAST_OP=34,
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
