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

#ifndef COSC_4P80_ASSIGNMENT_3_ARRAY_H
#define COSC_4P80_ASSIGNMENT_3_ARRAY_H

#include <assign3/fwdecl.h>
#include <assign3/neuron.h>

namespace assign3
{
    
    class array_t
    {
        public:
            explicit array_t(blt::size_t dimensions, blt::size_t width, blt::size_t height):
                    width(static_cast<blt::i64>(width)), height(static_cast<blt::i64>(height))
            {
                for (blt::size_t i = 0; i < width; i++)
                    for (blt::size_t j = 0; j < height; j++)
                        map.emplace_back(dimensions, i, j);
            }
            
            inline neuron_t& get(blt::size_t x, blt::size_t y)
            { return map[y * width + x]; }
            
            [[nodiscard]] inline const neuron_t& get(blt::size_t x, blt::size_t y) const
            { return map[y * width + x]; }
            
            [[nodiscard]] inline blt::size_t get_width() const
            { return width; }
            
            [[nodiscard]] inline blt::size_t get_height() const
            { return height; }
            
            [[nodiscard]] inline std::vector<neuron_t>& get_map()
            { return map; }
            
            [[nodiscard]] inline const std::vector<neuron_t>& get_map() const
            { return map; }
        
        private:
            [[nodiscard]] inline blt::i64 wrap_width(blt::i64 x) const;
            
            [[nodiscard]] inline blt::i64 wrap_height(blt::i64 y) const;
        
        private:
            blt::i64 width, height;
            std::vector<neuron_t> map;
    };
    
    
}

#endif //COSC_4P80_ASSIGNMENT_3_ARRAY_H
