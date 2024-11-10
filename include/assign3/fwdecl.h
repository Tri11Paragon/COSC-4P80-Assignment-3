#pragma once
/*
 *  Copyright (C) 2024  Brett Terpstra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef COSC_4P80_ASSIGNMENT_3_FWDECL_H
#define COSC_4P80_ASSIGNMENT_3_FWDECL_H

#include <blt/std/types.h>

namespace assign3
{
    using Scalar = float;
    
    inline constexpr blt::i32 RENDER_2D = 0x0;
    inline constexpr blt::i32 RENDER_3D = 0x1;
    
    enum class shape : blt::i32
    {
        GRID = RENDER_2D,
        GRID_WRAP = RENDER_2D,
        GRID_OFFSET = RENDER_2D,
        GRID_OFFSET_WRAP = RENDER_2D,
        GAUSSIAN_DIST = RENDER_2D,
        TOROIDAL = RENDER_3D,
        CYLINDER = RENDER_3D
    };
}

#endif //COSC_4P80_ASSIGNMENT_3_FWDECL_H
