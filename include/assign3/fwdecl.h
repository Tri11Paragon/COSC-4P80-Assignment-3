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
#include <blt/std/hashmap.h>
#include <array>

namespace assign3
{
    using Scalar = float;
    
    inline constexpr blt::i32 RENDER_2D = 0x0;
    inline constexpr blt::i32 RENDER_3D = 0x1;
    
    enum class shape_t : blt::i32
    {
        GRID,
        GRID_WRAP,
        GRID_OFFSET,
        GRID_OFFSET_WRAP
    };
    
    inline std::array<std::string, 4> shape_names{
            "Grid",
            "Edge Wrapped Grid",
            "Honey Comb Grid",
            "Edge Wrapped Honey Comb"
    };
    
    enum class debug_t
    {
        DATA_POINT,
        DISTANCE
    };
    
    inline std::array<std::string, 2> debug_names{
            "Distance to Datapoint",
            "Distance to Neighbours"
    };
    
    enum class init_t
    {
        COMPLETELY_RANDOM,
        RANDOM_DATA,
        SAMPLED_DATA
    };
    
    inline std::array<std::string, 3> init_names{
            "Random Unit",
            "Random Bounded",
            "Random Sample"
    };
    
    inline std::array<std::string, 3> init_helps{
            "Initializes weights randomly between -1 and 1",
            "Find min and max of each data element, then initialize weights between that range",
            "Initialize weights based on the input data"
    };
}

#endif //COSC_4P80_ASSIGNMENT_3_FWDECL_H
