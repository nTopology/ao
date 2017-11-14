#include <unordered_map>

#include "ao/eval/tape.hpp"

namespace Kernel {

Tape::Tape(const Tree root)
{
    auto flat = root.ordered();

    // Helper function to create a new clause in the data array
    // The dummy clause (0) is mapped to the first result slot
    std::unordered_map<Tree::Id, Clause::Id> clauses = {{nullptr, 0}};
    Clause::Id id = flat.size();

    // Helper function to make a new function
    std::list<Clause> tape_;
    auto newClause = [&clauses, &id, &tape_](const Tree::Id t)
    {
        tape_.push_front(
                {t->op,
                 id,
                 clauses.at(t->lhs.get()),
                 clauses.at(t->rhs.get())});
    };

    // Write the flattened tree into the tape!
    for (const auto& m : flat)
    {
        // Normal clauses end up in the tape
        if (m->rank > 0)
        {
            newClause(m.id());
        }
        // For constants and variables, record their values so
        // that we can store those values in the result array
        else if (m->op == Opcode::CONST)
        {
            constants[id] = m->value;
        }
        else if (m->op == Opcode::VAR)
        {
          vars.left.insert({ id, m.id() });
        }
        // For primitives, record their pointers so that
        // they can be used when calculating a point.
        else if (m->op == Opcode::PRIMITIVE) {
          assert(m->prim); //Should not be a null pointer.
          primitives[id] = m->prim;
        }
        else
        {
            assert(m->op == Opcode::VAR_X ||
                   m->op == Opcode::VAR_Y ||
                   m->op == Opcode::VAR_Z);
        }
        clauses[m.id()] = id--;
    }
    assert(id == 0);

    //  Move from the list tape to a more-compact vector tape
    tapes.push_back(Subtape());
    tape = tapes.begin();
    for (auto& t : tape_)
    {
        tape->t.push_back(t);
    }

    // With primitives, we no longer make sure that X, Y, Z have been allocated space;
    // any primitives not in the tree don't need space.  We still record the locations
    // of the old-style axes if they exist.
    std::vector<Tree> axes = {Tree::XOpc(), Tree::YOpc(), Tree::ZOpc()};
    /*for (auto a : axes)
    {
        if (clauses.find(a.id()) == clauses.end())
        {
            clauses[a.id()] = clauses.size();
        }
    }*/

    // Store the total number of clauses
    // Remember, evaluators need to allocate one more than this
    // amount of space, as the clause with id = 0 is a placeholder
    num_clauses = clauses.size() - 1;

    // Allocate enough memory for all the clauses
    disabled.resize(clauses.size());
    remap.resize(clauses.size());

    // Save old-style X, Y, Z ids, or 0 if they don't exist in the tree.
    XOpc = clauses.find(axes[0].id()) != clauses.end() ? clauses.at(axes[0].id()) : 0;
    YOpc = clauses.find(axes[1].id()) != clauses.end() ? clauses.at(axes[0].id()) : 0;
    ZOpc = clauses.find(axes[2].id()) != clauses.end() ? clauses.at(axes[0].id()) : 0;

    // Store the index of the tree's root
    assert(clauses.at(root.id()) == 1);
    tape->i = clauses.at(root.id());
};

void Tape::pop()
{
    assert(tape != tapes.begin());
    tape--;
}

double Tape::utilization() const
{
    return tape->t.size() / double(tapes.front().t.size());
}


Clause::Id Tape::rwalk(std::function<void(Opcode::Opcode, Clause::Id,
                                          Clause::Id, Clause::Id)> fn,
                       bool& abort)
{
    for (auto itr = tape->t.rbegin(); itr != tape->t.rend() && !abort; ++itr)
    {
        fn(itr->op, itr->id, itr->a, itr->b);
    }
    return tape->i;
}

void Tape::walk(std::function<void(Opcode::Opcode, Clause::Id,
                                   Clause::Id, Clause::Id)> fn, bool& abort)
{
    for (auto itr = tape->t.begin(); itr != tape->t.end() && !abort; ++itr)
    {
        fn(itr->op, itr->id, itr->a, itr->b);
    }
}

void Tape::push(std::function<Keep(Opcode::Opcode, Clause::Id,
                                   Clause::Id, Clause::Id)> fn,
                Type t, Region<3> r)
{
    // Since we'll be figuring out which clauses are disabled and
    // which should be remapped, we reset those arrays here
    std::fill(disabled.begin(), disabled.end(), true);
    std::fill(remap.begin(), remap.end(), 0);

    // Mark the root node as active
    disabled[tape->i] = false;

    for (const auto& c : tape->t)
    {
        if (!disabled[c.id])
        {
            switch (fn(c.op, c.id, c.a, c.b))
            {
                case KEEP_A:    disabled[c.a] = false;
                                remap[c.id] = c.a;
                                break;
                case KEEP_B:    disabled[c.b] = false;
                                remap[c.id] = c.b;
                                break;
                case KEEP_BOTH: break;

            }

            if (!remap[c.id])
            {
                disabled[c.a] = false;
                disabled[c.b] = false;
            }
            else
            {
                disabled[c.id] = true;
            }
        }
    }

    auto prev_tape = tape;

    // Add another tape to the top of the tape stack if one doesn't already
    // exist (we never erase them, to avoid re-allocating memory during
    // nested evaluations).
    if (++tape == tapes.end())
    {
        tape = tapes.insert(tape, Subtape());
        tape->t.reserve(tapes.front().t.size());
    }
    else
    {
        // We may be reusing an existing tape, so resize to 0
        // (preserving allocated storage)
        tape->t.clear();
    }

    assert(tape != tapes.end());
    assert(tape != tapes.begin());
    assert(tape->t.capacity() >= prev_tape->t.size());

    // Reset tape type
    tape->type = t;

    // Now, use the data in disabled and remap to make the new tape
    for (const auto& c : prev_tape->t)
    {
        if (!disabled[c.id])
        {
            Clause::Id ra, rb;
            for (ra = c.a; remap[ra]; ra = remap[ra]);
            for (rb = c.b; remap[rb]; rb = remap[rb]);
            tape->t.push_back({c.op, c.id, ra, rb});
        }
    }

    // Remap the tape root index
    for (tape->i = prev_tape->i; remap[tape->i]; tape->i = remap[tape->i]);

    // Make sure that the tape got shorter
    assert(tape->t.size() <= prev_tape->t.size());

    // Store X / Y / Z bounds (may be irrelevant)
    tape->X = {r.lower.x(), r.upper.x()};
    tape->Y = {r.lower.y(), r.upper.y()};
    tape->Z = {r.lower.z(), r.upper.z()};
}

}   // namespace Kernel
