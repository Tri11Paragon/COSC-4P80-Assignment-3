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
#include <blt/std/assert.h>

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
    
    Scalar toroidal_euclidean_distance_function_t::distance(blt::span<const Scalar> x, blt::span<const Scalar> y) const
    {
        BLT_ASSERT(x.size() == 2 && y.size() == 2);
        Scalar x_diff = std::abs(x[0] - y[0]);
        Scalar y_diff = std::abs(x[1] - y[1]);
        Scalar x_min = std::min(x_diff, static_cast<Scalar>(width) - x_diff);
        Scalar y_min = std::min(y_diff, static_cast<Scalar>(height) - y_diff);
        return std::sqrt(x_min * x_min + y_min * y_min);
    }
    
    Scalar axial_distance_function_t::distance(blt::span<const Scalar> x, blt::span<const Scalar> y) const
    {
        static thread_local std::vector<Scalar> distances;
        distances.clear();
        Scalar total = 0;
        for (auto [q, r] : blt::in_pairs(x, y))
        {
            distances.push_back(std::abs(q - r));
            total += distances.back();
        }
        
        Scalar min = distances.front();
        for (auto v : distances)
            min = std::min(min, v);

        return total - min;
    }
    
    Scalar axial_distance(Scalar q1, Scalar r1, Scalar q2, Scalar r2) {
        return (std::abs(q1 - q2) + std::abs(r1 - r2) + std::abs((q1 + r1) - (q2 + r2))) / 2;
    }
    
    Scalar toroidal_axial_distance_function_t::distance(blt::span<const Scalar> x, blt::span<const Scalar> y) const
    {
        BLT_ASSERT(x.size() == 2 && y.size() == 2);
        
        Scalar x_diff = std::abs(x[0] - y[0]);
        Scalar y_diff = std::abs(x[1] - y[1]);
        Scalar x_min = std::min(x_diff, static_cast<Scalar>(width) - x_diff);
        Scalar y_min = std::min(y_diff, static_cast<Scalar>(height) - y_diff);
        Scalar total = x_min + y_min;
        return total - std::min(x_min, y_min);
    }
    
    std::unique_ptr<distance_function_t> distance_function_t::from_shape(shape_t shape, blt::u32 som_width, blt::u32 som_height)
    {
        switch (shape)
        {
            case shape_t::GRID:
                return std::make_unique<euclidean_distance_function_t>();
            case shape_t::GRID_WRAP:
                return std::make_unique<toroidal_euclidean_distance_function_t>(som_width, som_height);
            case shape_t::GRID_OFFSET:
                return std::make_unique<axial_distance_function_t>();
            case shape_t::GRID_OFFSET_WRAP:
                return std::make_unique<toroidal_axial_distance_function_t>(som_width, som_height);
        }
        return nullptr;
    }
}