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
#include <assign3/functions.h>
#include <cmath>
#include "blt/iterator/zip.h"

namespace assign3
{
    
    Scalar gaussian_function_t::call(Scalar dist, Scalar r) const
    {
        auto dist_sq = dist * dist;
        return std::exp(-r * dist_sq);
    }
    
    Scalar gaussian_function_t::scale(Scalar half_distance, Scalar target_strength) const
    {
        return -std::log(target_strength) / (half_distance * half_distance);
    }
    
    Scalar euclidean_distance_function_t::distance(blt::span<const Scalar> x, blt::span<const Scalar> y) const
    {
        Scalar dist = 0;
        for (auto [a, b] : blt::in_pairs(x, y))
        {
            auto d = a - b;
            dist += d * d;
        }
        return std::sqrt(dist);
    }
}