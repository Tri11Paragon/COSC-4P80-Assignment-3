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
#include <assign3/neuron.h>
#include <blt/std/random.h>
#include <blt/iterator/iterator.h>
#include <cmath>
#include "blt/std/logging.h"

namespace assign3
{
    neuron_t& neuron_t::randomize(blt::size_t seed)
    {
        blt::random::random_t rand{seed};
        
        for (auto& v : data)
            v = static_cast<Scalar>(rand.get_double(-1, 1));
        
        return *this;
    }
    
    // apply the distance based on the update neuron function
    neuron_t& neuron_t::update(const std::vector<Scalar>& new_data, Scalar dist, Scalar eta)
    {
        static thread_local std::vector<Scalar> diff;
        diff.clear();
        
        for (auto [v, x] : blt::in_pairs(data, new_data))
            diff.push_back(x - v);
        
        for (auto [v, d] : blt::in_pairs(data, diff))
            v += eta * dist * d;
        
        return *this;
    }
    
    // distance between an input vector and the neuron, in the n-space
    Scalar neuron_t::dist(const std::vector<Scalar>& X) const
    {
        euclidean_distance_function_t dist_func;
        return dist_func.distance(data, X);
    }
    
    // distance between two neurons, in 2d
    Scalar neuron_t::distance(const neuron_t& n1, const neuron_t& n2)
    {
        euclidean_distance_function_t dist_func;
        return dist_func.distance({n1.get_x(), n1.get_y()}, {n2.get_x(), n2.get_y()});
    }
}