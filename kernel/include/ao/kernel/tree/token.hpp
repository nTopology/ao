/*
 *  Copyright (C) 2016 Matthew Keeter  <matt.j.keeter@gmail.com>
 *
 *  This file is part of the Ao library.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  Ao is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with Ao.  If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include <cstdlib>
#include <memory>

#include "glm/vec4.hpp"

#include "ao/kernel/tree/opcode.hpp"
#include "ao/kernel/tree/cache.hpp"

/*
 *  A token represents a single expression (with up to two arguments)
 */
class Token
{
public:
    /*
     *  Returns a token for the given constant
     */
    static Token constant(float v);

    /*
     *  Returns a token for the given operation
     *
     *  Arguments should be filled in from left to right
     *  (i.e. a must not be null if b is not null)
     */
    static Token operation(Opcode::Opcode op,
                           Token a=Token(0, nullptr),
                           Token b=Token(0, nullptr));

    /*
     *  Constructors for individual axes (non-affine)
     *
     *  The three-axis constructor axes() should be preferred to using
     *  an individual-axis constructor, as it ensures that they're backed
     *  by the same Cache object
     */
    static Token X();
    static Token Y();
    static Token Z();

    /*
     *  Returns an AFFINE token (of the form a*x + b*y + c*z + d)
     */
    static Token affine(float a, float b, float c, float d);

    /*
     *  Returns a tuple of affine tokens for X, Y, Z axes
     */
    static std::tuple<Token, Token, Token> axes();

    /*
     *  Returns a new Token that is a flattened copy of this tree
     */
    Token collapse() const;

    /*
     *  Attempts to get affine terms from a AFFINE_VEC token
     *  If success is provided, it is populated with true or false
     */
    glm::vec4 getAffine(bool* success=nullptr);

    /*
     *  Accessors for token fields
     */
    Opcode::Opcode opcode() const;
    Token lhs() const;
    Token rhs() const;
    size_t rank() const;
    float value() const;

    /*  ID indexing into the parent Cache */
    const Cache::Id id;

protected:
    /*
     *  Private constructor
     *  (only ever called by Cache)
     */
    explicit Token(Cache::Id id, Cache* parent) : id(id), parent(parent) {}

    /*  Shared pointer to parent Cache
     *  Every token that refers back to this cache has a pointer to it,
     *  and the Cache is only deleted when all tokens are destroyed     */
    std::shared_ptr<Cache> parent;

    /*  An Evaluator needs to be able to pull out the parent Cache */
    friend class Evaluator;
};

