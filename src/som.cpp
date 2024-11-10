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
#include <assign3/som.h>
#include <random>
#include <algorithm>
#include <blt/std/random.h>
#include <blt/iterator/enumerate.h>
#include <blt/std/logging.h>
#include <cstring>

namespace assign3
{
    
    som_t::som_t(const data_file_t& file, blt::size_t width, blt::size_t height, blt::size_t max_epochs):
            array(file.data_points.begin()->bins.size(), width, height), file(file), max_epochs(max_epochs)
    {
        for (auto& v : array.get_map())
            v.randomize(std::random_device{}());
    }
    
    void som_t::train_epoch(Scalar initial_learn_rate, topology_function_t* basis_func)
    {
        blt::random::random_t rand{std::random_device{}()};
        std::shuffle(file.data_points.begin(), file.data_points.end(), rand);
        
        auto time_ratio = static_cast<Scalar>(current_epoch) / static_cast<Scalar>(max_epochs);
        auto eta = initial_learn_rate * std::exp(-2 * time_ratio);
        
        for (auto& current_data : file.data_points)
        {
            auto v0_idx = get_closest_neuron(current_data.bins);
            auto v0 = array.get_map()[v0_idx];
            v0.update(current_data.bins, 1, eta);
            
            // find the closest neighbour neuron to v0
            auto distance_min = find_closest_neighbour_distance(v0_idx);
            // this will find the required scaling factor to make a point in the middle between v0 and its closest neighbour activate 50%
            // from the perspective of the gaussian function
            auto scale = basis_func->scale(distance_min * 0.5f, 0.5);
            
            for (auto [i, n] : blt::enumerate(array.get_map()))
            {
                if (i == v0_idx)
                    continue;
                auto dist = basis_func->call(neuron_t::distance(v0, n), time_ratio * scale);
                n.update(current_data.bins, dist, eta);
            }
        }
        
        current_epoch++;
    }
    
    blt::size_t som_t::get_closest_neuron(const std::vector<Scalar>& data)
    {
        blt::size_t index = 0;
        Scalar distance = std::numeric_limits<Scalar>::max();
        for (auto [i, d] : blt::enumerate(array.get_map()))
        {
            auto dist = d.dist(data);
            if (dist < distance)
            {
                index = i;
                distance = dist;
            }
        }
        return index;
    }
    
    Scalar som_t::find_closest_neighbour_distance(blt::size_t v0)
    {
        Scalar distance_min = std::numeric_limits<Scalar>::max();
        for (const auto& [i, n] : blt::enumerate(array.get_map()))
        {
            if (i != v0)
                distance_min = std::min(distance_min, neuron_t::distance(array.get_map()[v0], n));
        }
        return distance_min;
    }
    
    
}