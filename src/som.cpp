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
    
    som_t::som_t(const data_file_t& file, blt::size_t width, blt::size_t height, blt::size_t max_epochs, distance_function_t* dist_func,
                 shape_t shape, init_t init, bool normalize):
            array(file.data_points.begin()->bins.size(), width, height, shape), file(file), max_epochs(max_epochs), dist_func(dist_func)
    {
        for (auto& v : array.get_map())
            v.randomize(std::random_device{}(), init, normalize, file);
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
                auto dist = basis_func->call(neuron_t::distance(dist_func, v0, n), time_ratio * scale);
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
                distance_min = std::min(distance_min, neuron_t::distance(dist_func, array.get_map()[v0], n));
        }
        return distance_min;
    }
    
    struct distance_data_t
    {
        Scalar data;
        blt::size_t index;
        
        distance_data_t(Scalar data, size_t index): data(data), index(index)
        {}
        
        inline friend bool operator<(const distance_data_t& a, const distance_data_t& b)
        {
            return a.data < b.data;
        }
        
        inline friend bool operator==(const distance_data_t& a, const distance_data_t& b)
        {
            return a.data == b.data;
        }
    };
    
    blt::vec2 som_t::get_topological_position(const std::vector<Scalar>& data)
    {
        std::vector<distance_data_t> distances;
        for (auto [i, d] : blt::enumerate(get_array().get_map()))
            distances.emplace_back(d.dist(data), i);
        std::sort(distances.begin(), distances.end());
        
        auto [dist_1, ni_1] = distances[0];
        auto [dist_2, ni_2] = distances[1];
        auto [dist_3, ni_3] = distances[2];
        
        float dt = dist_1 + dist_2 + dist_3;
        float dp1 = dist_1 / dt;
        float dp2 = dist_2 / dt;
        float dp3 = dist_3 / dt;
        
        auto n_1 = array.get_map()[ni_1];
        auto n_2 = array.get_map()[ni_2];
        auto n_3 = array.get_map()[ni_3];
        
        auto p_1 = blt::vec2{n_1.get_x(), n_1.get_y()};
        auto p_2 = blt::vec2{n_2.get_x(), n_2.get_y()};
        auto p_3 = blt::vec2{n_3.get_x(), n_3.get_y()};
        
        return (dp1 * p_1) + (dp2 * p_2) + (dp3 * p_3);
    }
    
    Scalar som_t::topological_error(const data_file_t& data)
    {
        Scalar total = 0;
        std::vector<std::pair<blt::size_t, Scalar>> distances;
        
        for (const auto& x : data.data_points)
        {
            distances.clear();
            for (const auto& [i, n] : blt::enumerate(array.get_map()))
                distances.emplace_back(i, n.dist(x.bins));
            
            std::pair<blt::size_t, Scalar> min1 = {0, std::numeric_limits<Scalar>::max()};
            std::pair<blt::size_t, Scalar> min2 = {0, std::numeric_limits<Scalar>::max()};
            
            for (const auto& elem : distances)
            {
                if (elem.second < min1.second)
                {
                    min2 = min1;
                    min1 = elem;
                } else if (elem.second < min2.second)
                    min2 = elem;
            }
            
            // we can assert the neurons are neighbours if the distance between the BMUs and the nearest neighbour are equal.
            auto min_distances = neuron_t::distance(dist_func, array.get_map()[min1.first], array.get_map()[min2.first]);
            auto neighbour_distances = find_closest_neighbour_distance(min1.first);
            
            if (!blt::f_equal(min_distances, neighbour_distances))
                total += 1;
        }
        
        return total / static_cast<Scalar>(data.data_points.size());
    }
    
    
}