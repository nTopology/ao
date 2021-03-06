/*
Studio: a simple GUI for the libfive CAD kernel
Copyright (C) 2017  Matt Keeter

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#pragma once

#include <QOpenGLShaderProgram>

namespace Shader
{
    void initializeGL();

    extern QOpenGLShaderProgram* flat;
    extern QOpenGLShaderProgram* monochrome;
    extern QOpenGLShaderProgram* shaded;
    extern QOpenGLShaderProgram* busy;
    extern QOpenGLShaderProgram* bars;
    extern QOpenGLShaderProgram* point;
    extern QOpenGLShaderProgram* line;
    extern QOpenGLShaderProgram* arrow;
}
