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

namespace assign3
{
    neuron_t& neuron_t::randomize(blt::size_t seed)
    {
        blt::random::random_t rand{seed};
        
        for (auto& v : data)
            v = static_cast<Scalar>(rand.get_double(-1, 1));
        
        return *this;
    }
    
    neuron_t& neuron_t::update(const std::vector<Scalar>& new_data, const topology_function_t* basis_func, Scalar eta, Scalar r)
    {
        auto d = dist(new_data);
        for (auto& v : data)
            v = eta * basis_func->call(d, r);
        return *this;
    }
    
    Scalar neuron_t::dist(const std::vector<Scalar>& X) const
    {
        Scalar dist = 0;
        for (auto [v, x] : blt::zip(data, X))
        {
            auto loc = (v - x);
            dist += loc * loc;
        }
        return std::sqrt(dist);
    }
    
    Scalar neuron_t::distance(const neuron_t& n1, const neuron_t& n2, Scalar time_ratio)
    {
        auto dist = n1.dist(n2.data);
        auto dist_sq = dist * dist;
        return std::exp(-time_ratio * dist_sq);
    }
}