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

#ifndef COSC_4P80_ASSIGNMENT_3_FUNCTIONS_H
#define COSC_4P80_ASSIGNMENT_3_FUNCTIONS_H

#include <assign3/fwdecl.h>
#include <blt/std/ranges.h>

namespace assign3
{
    
    struct topology_function_t
    {
        /**
         * @param dist input - usually the distance
         * @param r time ratio - t / max_t
         * @return basis results
         */
        [[nodiscard]] virtual Scalar call(Scalar dist, Scalar r) const = 0;
        
        /**
         * produces a non-time scaled r value for making the function activate at target_strength when a point is half_distance between two nodes
         * so for the example of unit distance 1, half distance is 0.5, target_strength is 0.5
         * this function will produce a value which can be time scaled for achieving this result at t=0
         *
         * this function can return a 1 to indicate this function doesn't care for this.
         */
        [[nodiscard]] virtual Scalar scale(Scalar half_distance, Scalar target_strength) const = 0;
        
        virtual ~topology_function_t() = default;
    };
    
    struct gaussian_function_t final : public topology_function_t
    {
        [[nodiscard]] Scalar call(Scalar dist, Scalar r) const final;
        
        [[nodiscard]] Scalar scale(Scalar half_distance, Scalar target_strength) const final;
    };
    
    struct distance_function_t
    {
        [[nodiscard]] virtual Scalar distance(blt::span<const Scalar> x, blt::span<const Scalar> y) const = 0;
        
        virtual ~distance_function_t() = default;
    };
    
    struct euclidean_distance_function_t : public distance_function_t
    {
        [[nodiscard]] Scalar distance(blt::span<const Scalar> x, blt::span<const Scalar> y) const final;
    };
    
}

#endif //COSC_4P80_ASSIGNMENT_3_FUNCTIONS_H
