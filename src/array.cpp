/*
 *  <Short Description>
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
#include <assign3/array.h>
#include <cmath>

namespace assign3
{
    
    blt::i64 array_t::wrap_height(blt::i64 y) const
    {
        if (y >= height)
            return y - height;
        else if (y < 0)
            return height + y;
        else
            return y;
    }
    
    blt::i64 array_t::wrap_width(blt::i64 x) const
    {
        if (x >= width)
            return x - width;
        else if (x < 0)
            return width + x;
        else
            return x;
    }
}