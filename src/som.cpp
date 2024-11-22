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
#include "blt/iterator/zip.h"

namespace assign3
{
    
    som_t::som_t(const data_file_t& file, blt::size_t width, blt::size_t height, blt::size_t max_epochs, distance_function_t* dist_func,
                 topology_function_t* topology_function, shape_t shape, init_t init, bool normalize):
            array(file.data_points.begin()->bins.size(), width, height, shape), file(file), max_epochs(max_epochs), dist_func(dist_func),
            topology_function(topology_function)
    {
        for (auto& v : array.get_map())
            v.randomize(std::random_device{}(), init, normalize, file);
        compute_errors();
    }
    
    void som_t::train_epoch(Scalar initial_learn_rate)
    {
        blt::random::random_t rand{std::random_device{}()};
        std::shuffle(file.data_points.begin(), file.data_points.end(), rand);
        
        auto time_ratio = static_cast<Scalar>(current_epoch) / static_cast<Scalar>(max_epochs);
        auto eta = initial_learn_rate * std::exp(-2 * time_ratio);
        
        for (auto& current_data : file.data_points)
        {
            const auto v0_idx = get_closest_neuron(current_data.bins);
            auto& v0 = array.get_map()[v0_idx];
            v0.update(current_data.bins, v0.dist(current_data.bins), eta);
            
            // find the closest neighbour neuron to v0
            const auto distance_min = find_closest_neighbour_distance(v0_idx);
            // this will find the required scaling factor to make a point in the middle between v0 and its closest neighbour activate 50%
            // from the perspective of the gaussian function
            const auto scale = topology_function->scale(distance_min * 0.5f, 0.5);
            
            for (auto [i, n] : blt::enumerate(array.get_map()))
            {
                if (i == v0_idx)
                    continue;
                auto dist = topology_function->call(neuron_t::distance(dist_func, v0, n), time_ratio * scale);
                n.update(current_data.bins, dist, eta);
            }
        }
        current_epoch++;
        compute_errors();
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
        
        const auto [dist_1, ni_1] = distances[0];
        const auto [dist_2, ni_2] = distances[1];
        const auto [dist_3, ni_3] = distances[2];
        
        const float dt = dist_1 + dist_2 + dist_3;
        const float dp1 = dist_1 / dt;
        const float dp2 = dist_2 / dt;
        const float dp3 = dist_3 / dt;
        
        const auto& n_1 = array.get_map()[ni_1];
        const auto& n_2 = array.get_map()[ni_2];
        const auto& n_3 = array.get_map()[ni_3];
        
        const auto p_1 = blt::vec2{n_1.get_x(), n_1.get_y()};
        const auto p_2 = blt::vec2{n_2.get_x(), n_2.get_y()};
        const auto p_3 = blt::vec2{n_3.get_x(), n_3.get_y()};
        
        return (dp1 * p_1) + (dp2 * p_2) + (dp3 * p_3);
    }
    
    Scalar som_t::topological_error()
    {
        Scalar total = 0;
        std::vector<std::pair<blt::size_t, Scalar>> distances;
        
        for (const auto& x : file.data_points)
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
            const auto min_distances = neuron_t::distance(dist_func, array.get_map()[min1.first], array.get_map()[min2.first]);
            auto neighbour_distances = find_closest_neighbour_distance(min1.first);
            
            if (!blt::f_equal(min_distances, neighbour_distances))
                total += 1;
        }
        
        return total / static_cast<Scalar>(file.data_points.size());
    }
    
    void som_t::compute_neuron_activations(Scalar distance, Scalar activation)
    {
        for (auto& n : array.get_map())
            n.set_activation(0);
        
        Scalar min = std::numeric_limits<Scalar>::max();
        Scalar max = std::numeric_limits<Scalar>::min();
        
        for (auto [i, v] : blt::enumerate(array.get_map()))
        {
            const auto half = find_closest_neighbour_distance(i) / distance;
//            auto sigma = std::sqrt(-(half * half) / (2 * std::log(requested_activation)));
//            auto r = 1 / (2 * sigma * sigma);
//
            const auto scale = topology_function->scale(half, activation);
            for (const auto& [is_bad, bins] : file.data_points)
            {
                const auto ds = topology_function->call(v.dist(bins), scale);
                if (is_bad)
                    v.activate(-ds);
                else
                    v.activate(ds);
            }
            
            min = std::min(min, v.get_activation());
            max = std::max(max, v.get_activation());
        }

        for (auto& n : array.get_map())
            n.set_activation(2 * (n.get_activation() - min) / (max - min) - 1);
    }
    
    void som_t::write_activations(std::ostream& out)
    {
        out << "x,y,activation\n";
        for (const auto& v : array.get_map())
            out << v.get_x() << ',' << v.get_y() << ',' << v.get_activation() << '\n';
    }
    
    void som_t::write_topology_errors(std::ostream& out)
    {
        out << "epoch,error\n";
        for (auto [i, v] : blt::enumerate(topological_errors))
            out << i << ',' << v << '\n';
    }
    
    void som_t::write_quantization_errors(std::ostream& out)
    {
        out << "epoch,error\n";
        for (auto [i, v] : blt::enumerate(quantization_errors))
            out << i << ',' << v << '\n';
    }
    
    void som_t::write_all_errors(std::ostream& out)
    {
        out << "epoch,topology error,quantization error\n";
        for (auto [i, v] : blt::in_pairs(topological_errors, quantization_errors).enumerate())
        {
            auto [t, q] = v;
            out << i << ',' << t << ',' << q << '\n';
        }
    }
    
    Scalar som_t::quantization_error()
    {
        Scalar incorrect = 0;
        
        for (const auto& point : file.data_points)
        {
            const auto& nearest = array.get_map()[get_closest_neuron(point.bins)];
            
            bool is_neural = nearest.get_activation() > -quantization_distance && nearest.get_activation() < quantization_distance;
            
            if (is_neural)
            {
                incorrect++;
                continue;
            }
            
            bool is_bad = nearest.get_activation() <= -quantization_distance;
            bool is_good = nearest.get_activation() >= quantization_distance;
            
            if ((is_bad && point.is_bad) || (is_good && !point.is_bad))
                continue;
            incorrect++;
        }

        return incorrect;
    }
    
    void som_t::compute_errors()
    {
        compute_neuron_activations();
        topological_errors.push_back(topological_error());
        quantization_errors.push_back(quantization_error());
    }
    
    
}