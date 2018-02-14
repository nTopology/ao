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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <list>
#include <cmath>
#include <cassert>

#include "libfive/tree/cache.hpp"
#include "libfive/tree/template.hpp"
#include "libfive/eval/transformed_primitive.h"

namespace Kernel {

Tree::Tree()
    : ptr(nullptr)
{
    // Nothing to do here
}

Tree::Tree(float v)
    : ptr(Cache::instance()->constant(v))
{
    // Nothing to do here
}


Tree::Tree(const Primitive& prim)
  : ptr(Cache::instance()->primitive(prim))
{
  //Nothing to do here either.
}


Tree Tree::XPrim() { return Tree((Cache::XPrimRef())); }
Tree Tree::YPrim() { return Tree((Cache::YPrimRef())); }
Tree Tree::ZPrim() { return Tree((Cache::ZPrimRef())); }

Tree::Tree(Opcode::Opcode op, Tree a, Tree b)
    : ptr(Cache::instance()->operation(op, a.ptr, b.ptr))
{
    // Aggressive sanity-checking
    assert((Opcode::args(op) == 0 && a.ptr.get() == 0 && b.ptr.get() == 0) ||
           (Opcode::args(op) == 1 && a.ptr.get() != 0 && b.ptr.get() == 0) ||
           (Opcode::args(op) == 2 && a.ptr.get() != 0 && b.ptr.get() != 0));

    // POW only accepts integral values as its second argument
    if (op == Opcode::POW)
    {
        assert(b->op == Opcode::CONST &&
               b->value == std::round(b->value));
    }
    else if (op == Opcode::NTH_ROOT)
    {
        assert(b->op == Opcode::CONST &&
               b->value == std::round(b->value) &&
               b->value > 0);
    }
}

Tree Tree::var()
{
    return Tree(Cache::instance()->var());
}

Tree::Tree_::~Tree_()
{
    if (op == Opcode::CONST)
    {
        Cache::instance()->del(value);
    }
    else if (op == Opcode::PRIMITIVE)
    {
        Cache::instance()->del(*prim);
    }
    else if (op != Opcode::VAR)
    {
        Cache::instance()->del(op, lhs, rhs);
    }
}

std::list<Tree> Tree::ordered() const
{
    std::set<Id> found = {nullptr};
    std::list<std::shared_ptr<Tree_>> todo = { ptr };
    std::map<unsigned, std::list<std::shared_ptr<Tree_>>> ranks;

    while (todo.size())
    {
        auto t = todo.front();
        todo.pop_front();

        if (found.find(t.get()) == found.end())
        {
            todo.push_back(t->lhs);
            todo.push_back(t->rhs);
            found.insert(t.get());
            ranks[t->rank].push_back(t);
        }
    }

    std::list<Tree> out;
    for (auto& r : ranks)
    {
        for (auto& t : r.second)
        {
            out.push_back(Tree(t));
        }
    }
    return out;
}

void Tree::serialize(std::ostream& oss) const {
  oss << "tree";
  for (auto i : ordered()) {
    if (i->prim) {
      i->prim->serialize(oss);
    }
  }
  oss << "\n";
}

std::vector<uint8_t> Tree::serialize() const
{
    return Template(*this).serialize();
}

Tree Tree::deserialize(const std::vector<uint8_t>& data)
{
    return Template::deserialize(data).tree;
}

Tree Tree::load(const std::string& filename)
{
    std::ifstream in(filename, std::ios::in|std::ios::binary|std::ios::ate);
    if (in.is_open())
    {
        std::vector<uint8_t> data;
        data.resize(in.tellg());

        in.seekg(0, std::ios::beg);
        in.read((char*)&data[0], data.size());

        auto t = Template::deserialize(data);
        return t.tree;
    }
    return Tree();
}

Tree Tree::remap(Tree X_, Tree Y_, Tree Z_) const
{
    std::map<Tree::Id, std::shared_ptr<Tree_>> m = {
        {X().id(), X_.ptr}, {Y().id(), Y_.ptr}, {Z().id(), Z_.ptr}};
    auto primitives = Cache::instance()->getPrimitives();
    for (auto iter = primitives.begin(); iter != primitives.end(); ++iter) {
      m[iter->second.lock().get()] = Tree(TransformedPrimitive(*(iter->first), X_, Y_, Z_)).ptr;
      //If multiple transformations are used in succession, a potential speedup may be to combine them so that
      //there is never a transformed primitive whose underlying is another transformed primitive.
    }

    return remap(m);
}

Tree Tree::remap(std::map<Id, std::shared_ptr<Tree_>> m) const
{
    for (const auto& t : ordered())
    {
        if (Opcode::args(t->op) >= 1)
        {
            auto lhs = m.find(t->lhs.get());
            auto rhs = m.find(t->rhs.get());
            m.insert({t.id(), Cache::instance()->operation(t->op,
                        lhs == m.end() ? t->lhs : lhs->second,
                        rhs == m.end() ? t->rhs : rhs->second)});
        }
    }

    // If this Tree was remapped, then return it; otherwise return itself
    auto r = m.find(id());
    return r == m.end() ? *this : Tree(r->second);
}

////////////////////////////////////////////////////////////////////////////////

void Tree::Tree_::print(std::ostream& stream, Opcode::Opcode prev_op)
{
    const bool commutative = (prev_op == op);
    const int args = Opcode::args(op);

    if (!commutative)
    {
        switch (args)
        {
            case 2:
            case 1: stream << "(" <<  Opcode::toOpString(op) << " ";
                    break;
            case 0:
                if (op == Opcode::CONST)
                {
                    if (value == int(value))
                    {
                        stream << int(value);
                    }
                    else
                    {
                        stream << value;
                    }
                }
                else
                {
                    stream << Opcode::toOpString(op);
                }
                break;
            default:    assert(false);
        }
    }

    const auto op_ = Opcode::isCommutative(op) ? op : Opcode::INVALID;
    switch (args)
    {
        case 2:     lhs->print(stream, op_);
                    stream << " ";
                    rhs->print(stream, op_);
                    break;
        case 1:     lhs->print(stream, op_);
                    break;
        case 0:     break;
        default:    assert(false);
    }

    if (!commutative && args > 0)
    {
        stream <<")";
    }
}

}   // namespace Kernel

////////////////////////////////////////////////////////////////////////////////

// Mass-produce definitions for overloaded operations
#define OP_UNARY(name, opcode) \
Kernel::Tree name(const Kernel::Tree& a) { return Kernel::Tree(opcode, a); }
OP_UNARY(square,    Kernel::Opcode::SQUARE)
OP_UNARY(sqrt,      Kernel::Opcode::SQRT)
Kernel::Tree Kernel::Tree::operator-() const
    { return Kernel::Tree(Kernel::Opcode::NEG, *this); }
OP_UNARY(abs,       Kernel::Opcode::ABS)
OP_UNARY(sin,       Kernel::Opcode::SIN)
OP_UNARY(cos,       Kernel::Opcode::COS)
OP_UNARY(tan,       Kernel::Opcode::TAN)
OP_UNARY(asin,      Kernel::Opcode::ASIN)
OP_UNARY(acos,      Kernel::Opcode::ACOS)
OP_UNARY(atan,      Kernel::Opcode::ATAN)
OP_UNARY(log,       Kernel::Opcode::LOG)
OP_UNARY(exp,       Kernel::Opcode::EXP)
#undef OP_UNARY

#define OP_BINARY(name, opcode) \
Kernel::Tree name(const Kernel::Tree& a, const Kernel::Tree& b) \
    { return Kernel::Tree(opcode, a, b); }
OP_BINARY(operator+,    Kernel::Opcode::ADD)
OP_BINARY(operator*,    Kernel::Opcode::MUL)
OP_BINARY(min,          Kernel::Opcode::MIN)
OP_BINARY(max,          Kernel::Opcode::MAX)
OP_BINARY(operator-,    Kernel::Opcode::SUB)
OP_BINARY(operator/,    Kernel::Opcode::DIV)
OP_BINARY(atan2,        Kernel::Opcode::ATAN2)
OP_BINARY(pow,          Kernel::Opcode::POW)
OP_BINARY(nth_root,     Kernel::Opcode::NTH_ROOT)
OP_BINARY(mod,          Kernel::Opcode::MOD)
OP_BINARY(nanfill,      Kernel::Opcode::NANFILL)
OP_BINARY(useInterval,  Kernel::Opcode::USEINTERVAL)
#undef OP_BINARY


std::ostream& operator<<(std::ostream& stream, const Kernel::Tree& tree)
{
    tree->print(stream);
    return stream;
}
