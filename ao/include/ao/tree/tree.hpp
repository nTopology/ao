#pragma once

#include <memory>
#include <list>
#include <vector>
#include <map>

#include "ao/tree/opcode.hpp"
#include "ao/eval/primitive.h"

namespace Kernel {

/*
 *  A Tree represents a tree of math expressions.
 *
 *  A Tree is a lightweight wrapper containing a shared_ptr to a Tree_,
 *  where data is actually stored.  Deduplication is handled by a global
 *  Cache object
 */
class Tree
{
public:
    /*
     *  Returns a Tree for the given constant
     */
    Tree(float v);


    /*
    *  Returns a Tree for the given primitive
    */
    Tree(const Primitive& prim);

    /*
     *  Constructors for individual axes.
     */
    static Tree XOpc() { return Tree(Opcode::VAR_X); }
    static Tree YOpc() { return Tree(Opcode::VAR_Y); }
    static Tree ZOpc() { return Tree(Opcode::VAR_Z); }

    static Tree X();
    static Tree Y();
    static Tree Z();

    /*
     *  Used to mark a bad parse, among other things
     */
    static Tree Invalid() { return Tree(nullptr); }

    /*
     *  Returns a token for the given operation
     *
     *  Arguments should be filled in from left to right
     *  (i.e. a must not be null if b is not null)
     *
     *  If the opcode is POW or NTH_ROOT, b must be an integral CONST
     *  (otherwise an assertion will be triggered).
     *  If the opcode is NTH_ROOT, b must be > 0.
     */
    explicit Tree(Opcode::Opcode op, Tree a=Tree(), Tree b=Tree());

    /*
     *  Returns a new unique variable
     */
    static Tree var();

    /*  Bitfield enum for node flags */
    enum Flags {
        /*  Does this Id only contain constants and variables
         *  (no VAR_X, VAR_Y, VAR_Z, or PRIMITIVE opcodes allowed) */
        FLAG_LOCATION_AGNOSTIC  = (1<<1),
    };

    /*  This is where tree data is actually stored  */
    struct Tree_ {
        /*
         *  Destructor erases this Tree from the global Cache
         */

        ~Tree_();

        const Opcode::Opcode op;
        const uint8_t flags;
        const unsigned rank;

        /*  Only populated for constants  */
        const float value;

        /* Only populated for primitives */
        const Primitive* prim;

        /*  Only populated for operations  */
        const std::shared_ptr<Tree_> lhs;
        const std::shared_ptr<Tree_> rhs;

        /*
         *  Pushes a Scheme-format serialization to an ostream
         */
        void print(std::ostream& stream,
                   Opcode::Opcode prev_op=Opcode::INVALID);
    };

    /*  Trees are uniquely identified by their Tree_ address, but we don't
     *  want anyone to do anything with that value  */
    typedef const Tree_* Id;

    /*
     *  Overload arrow to get shared Tree_ value
     */
    const std::shared_ptr<Tree_>& operator->() const { return ptr; }

    /*
     *  Comparison operator for trees
     */
    bool operator==(const Tree& other) const {
        return ptr.get() == other.ptr.get();
    }

    /*
     *  Overloaded operators
     */
    Tree operator-() const;

    /*
     *  Unique identity (as the tree pointer)
     */
    Id id() const { return ptr.get(); }

    /*
     *  Remaps the base coordinates.  Does not affect primitives dependent on X, Y, and Z coordinates (even
     *  the primitive versions of those coordinates).
     */
    Tree remap(Tree X, Tree Y, Tree Z) const;

    /*
     *  Executes an arbitrary remapping
     */
    Tree remap(std::map<Id, std::shared_ptr<Tree_>> m) const;

    /*
     *  Walks the tree in rank order, from lowest to highest
     *  The last item in the list will be the tree this is called on
     */
    std::list<Tree> ordered() const;

    /*
     *  Serializes to a vector of bytes
     */
    std::vector<uint8_t> serialize() const;

    /*
     *  Deserialize a tree from a set of bytes
     */
    static Tree deserialize(const std::vector<uint8_t>& data);

    /*
     *  Loads a tree from a file
     */
    static Tree load(const std::string& filename);

protected:
    /*
     *  Empty tree constructor
     */
    explicit Tree();

    /*
     *  Private constructor
     */
    explicit Tree(std::shared_ptr<Tree_> t) : ptr(t) {}

    /*  Here's the actual Tree data */
    std::shared_ptr<Tree_> ptr;

    /*  These classes need access to private constructor  */
    friend class Cache;
};

}   // namespace Kernel

// Mass-produce declarations for overloaded operations
#define OP_UNARY(OP)      Kernel::Tree OP(const Kernel::Tree& a)
OP_UNARY(square);
OP_UNARY(sqrt);
OP_UNARY(abs);
OP_UNARY(sin);
OP_UNARY(cos);
OP_UNARY(tan);
OP_UNARY(asin);
OP_UNARY(acos);
OP_UNARY(atan);
OP_UNARY(exp);
#undef OP_UNARY

#define OP_BINARY(OP)     Kernel::Tree OP(const Kernel::Tree& a, const Kernel::Tree& b)
OP_BINARY(operator+);
OP_BINARY(operator*);
OP_BINARY(min);
OP_BINARY(max);
OP_BINARY(operator-);
OP_BINARY(operator/);
OP_BINARY(atan2);
OP_BINARY(pow);
OP_BINARY(nth_root);
OP_BINARY(mod);
OP_BINARY(nanfill);
OP_BINARY(cleanUnion);
OP_BINARY(cleanIntersect);
#undef OP_BINARY

/*
 *  Deserialize with Scheme-style syntax
 */
std::ostream& operator<<(std::ostream& stream, const Kernel::Tree& tree);