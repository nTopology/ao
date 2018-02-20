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

#include <map>
#include <memory>
#include <mutex>

#include "libfive/tree/tree.hpp"
#include "libfive/eval/primitive.h"

namespace Kernel {

/*
 *  A Cache stores values in a deduplicated math expression
 */
class Cache
{
    /*  Helper typedef to avoid writing this over and over again  */
    typedef std::shared_ptr<Tree::Tree_> Node;

    typedef std::tuple<Opcode::Opcode,  /* opcode */
                       Tree::Id,        /* lhs */
                       Tree::Id         /* rhs */ > Key;

    /*  Handle to safely access cache  */
    class Handle
    {
    public:
        Handle() : lock(mut) {}
        Cache* operator->() const { return &_instance; }
    protected:
        std::unique_lock<std::recursive_mutex> lock;
    };

public:
    /*
     *  Returns a safe (locking) handle to the global Cache
     */
    static Handle instance() { return Handle(); }

    Node constant(float v);
    Node operation(Opcode::Opcode op, Node lhs=nullptr, Node rhs=nullptr,
                   bool simplify=true);

    Node X() { return operation(Opcode::VAR_X); }
    Node Y() { return operation(Opcode::VAR_Y); }
    Node Z() { return operation(Opcode::VAR_Z); }

    Node XPrim() { return primitive(_XPos); }
    Node YPrim() { return primitive(_YPos); }
    Node ZPrim() { return primitive(_ZPos); }


    static const XPos& XPrimRef() { return _XPos; }
    static const YPos& YPrimRef() { return _YPos; }
    static const ZPos& ZPrimRef() { return _ZPos; }

    const std::map<float, std::weak_ptr<Tree::Tree_>> getConstants() const { return constants; }
    const std::map<Key, std::weak_ptr<Tree::Tree_>> getOps() const { return ops; }
    const std::map<const Primitive*, std::weak_ptr<Tree::Tree_>> getPrimitives() const { return primitives; }

    Node var();
    Node primitive(const Primitive& prim);

    /*
     *  Called when the last Tree_ is destroyed
     */
    void del(float v);
    void del(Opcode::Opcode op, Node lhs=nullptr, Node rhs=nullptr);
    void del(const Primitive& prim);

    /*
     *  Returns the given node as an affine sum-of-multiplications
     *
     *  This is a building block for automatic collapsing of affine
     *  expressions, exposed here primarily for unit testing.
     */
    std::map<Node, float> asAffine(Node n);

    /*
     *  Converts a sum-of-multiplications into an affine tree.
     */
    Node fromAffine(const std::map<Node, float>& ns);

protected:
    /*
     *  Cache constructor is private so outsiders must use instance()
     */
    Cache() {}

    /*
     *  Checks whether the operation is an identity operation
     *  If so returns an appropriately simplified tree
     *  i.e. (X + 0) will return X
     */
    Node checkIdentity(Opcode::Opcode op, Node a, Node b);

    /*
     *  If the opcode is commutative, consider tweaking tree structure
     *  to keep it as balanced as possible.
     */
    Node checkCommutative(Opcode::Opcode op, Node a, Node b);

    /*
     *  Checks whether a `op` b can be replaced by a simpler affine
     *  form, e.g. (2*x + y) - (4*y) --> 2*x - 3*y
     */
    Node checkAffine(Opcode::Opcode op, Node a, Node b);

    /*
     *  A Key uniquely identifies an operation Node, so that we can
     *  deduplicate based on opcode  and arguments
     */
    std::map<Key, std::weak_ptr<Tree::Tree_>> ops;

    /*  Constants in the tree are uniquely identified by their value  */
    std::map<float, std::weak_ptr<Tree::Tree_>> constants;

    /* And primitives by the pointer to them */
    std::map<const Primitive*, std::weak_ptr<Tree::Tree_>> primitives;

    static std::recursive_mutex mut;
    static Cache _instance;
    static XPos _XPos;
    static YPos _YPos;
    static ZPos _ZPos;
};

}   // namespace Kernel
