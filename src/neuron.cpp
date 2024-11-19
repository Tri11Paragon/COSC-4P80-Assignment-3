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
    neuron_t& neuron_t::randomize(blt::size_t seed, init_t init, bool normalize, const data_file_t& file)
    {
        blt::random::random_t rand{seed};
        switch (init)
        {
            case init_t::COMPLETELY_RANDOM:
                for (auto& v : data)
                    v = static_cast<Scalar>(rand.get_double(-1, 1));
                break;
            case init_t::RANDOM_DATA:
            {
                static thread_local std::vector<Scalar> min_values, max_values;
                min_values.clear();
                max_values.clear();
                
                min_values.resize(data.size());
                max_values.resize(data.size());
                
                for (const auto& [min, max] : blt::in_pairs(min_values, max_values))
                {
                    min = std::numeric_limits<Scalar>::max();
                    max = std::numeric_limits<Scalar>::min();
                }
                
                for (const auto& point : file.data_points)
                {
                    for (const auto& [i, bin] : blt::enumerate(point.bins))
                    {
                        min_values[i] = std::min(min_values[i], bin);
                        max_values[i] = std::max(max_values[i], bin);
                    }
                }
                for (const auto& [i, v] : blt::enumerate(data))
                    v = static_cast<Scalar>(rand.get_double(min_values[i], max_values[i]));
            }
                break;
            case init_t::SAMPLED_DATA:
            {
                auto selected = rand.select(file.data_points);
                std::memcpy(data.data(), selected.bins.data(), data.size() * sizeof(Scalar));
            }
                break;
        }
        
        if (normalize)
        {
            Scalar total = 0;
            for (auto v : data)
                total += v * v;
            Scalar mag = std::sqrt(total);
            for (auto& v : data)
                v /= mag;
        }
        
        return *this;
    }
    
    // apply the distance based on the update neuron function
    neuron_t& neuron_t::update(const std::vector<Scalar>& new_data, Scalar dist, Scalar eta)
    {
//        static thread_local std::vector<Scalar> diff;
//        diff.clear();

//        for (auto [v, x] : blt::in_pairs(data, new_data))
//            diff.push_back(x - v);
        
        for (auto [v, d] : blt::in_pairs(data, new_data))
            v += eta * dist * (d - v);
        
        return *this;
    }
    
    // distance between an input vector and the neuron, in the n-space
    Scalar neuron_t::dist(const std::vector<Scalar>& X) const
    {
        euclidean_distance_function_t dist_func;
        return dist_func.distance(data, X);
    }
    
    // distance between two neurons, in 2d
    Scalar neuron_t::distance(distance_function_t* dist_func, const neuron_t& n1, const neuron_t& n2)
    {
        return dist_func->distance({n1.get_x(), n1.get_y()}, {n2.get_x(), n2.get_y()});
    }
}