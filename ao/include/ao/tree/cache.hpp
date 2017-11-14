#pragma once

#include <map>
#include <memory>
#include <mutex>

#include "ao/tree/tree.hpp"
#include "ao/eval/primitive.h"

namespace Kernel {

/*
 *  A Cache stores values in a deduplicated math expression
 */
class Cache
{
    /*  Helper typedef to avoid writing this over and over again  */
    typedef std::shared_ptr<Tree::Tree_> Node;

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

    Node X() { return primitive(_XPos); }
    Node Y() { return primitive(_YPos); }
    Node Z() { return primitive(_ZPos); }

    static const XPos& XPrim() { return _XPos; }
    static const YPos& YPrim() { return _YPos; }
    static const ZPos& ZPrim() { return _ZPos; }

    std::map<const Primitive*, std::weak_ptr<Tree::Tree_>> getPrimitives() { return primitives; }

    Node var();
    Node primitive(const Primitive& prim);

    /*
     *  Called when the last Tree_ is destroyed
     */
    void del(float v);
    void del(Opcode::Opcode op, Node lhs=nullptr, Node rhs=nullptr);
    void del(const Primitive& prim);

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
     *  A Key uniquely identifies an operation Node, so that we can
     *  deduplicate based on opcode  and arguments
     */
    typedef std::tuple<Opcode::Opcode,  /* opcode */
                       Tree::Id,        /* lhs */
                       Tree::Id         /* rhs */ > Key;
    std::map<Key, std::weak_ptr<Tree::Tree_>> ops;

    /*  Constants in the tree are uniquely identified by their value  */
    std::map<float, std::weak_ptr<Tree::Tree_>> constants;

    /* And primitives by the pointer to them */
    std::map<const Primitive*, std::weak_ptr<Tree::Tree_>> primitives;

    //static variables implemented in the cpp are used to ensure one shared among all trees,
    //as avoiding deduplication is the whole point.  If it proves necessary to have multiple
    //independent trees, the cache can be changed to a property of the tree, and these variables
    //changed to normal ones.
    static std::recursive_mutex mut;
    static Cache _instance;
    static XPos _XPos;
    static YPos _YPos;
    static ZPos _ZPos;
};

}   // namespace Kernel
