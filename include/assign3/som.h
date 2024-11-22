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

#ifndef COSC_4P80_ASSIGNMENT_3_SOM_H
#define COSC_4P80_ASSIGNMENT_3_SOM_H

#include <assign3/array.h>
#include <assign3/file.h>
#include <assign3/functions.h>

namespace assign3
{
    class som_t
    {
    public:
        som_t(const data_file_t& file, blt::size_t width, blt::size_t height, blt::size_t max_epochs, distance_function_t* dist_func,
              topology_function_t* topology_function, shape_t shape, init_t init, bool normalize);

        som_t(const som_t&) = delete;
        som_t& operator=(const som_t&) = delete;
        som_t(som_t&&) = default;
        som_t& operator=(som_t&&) = default;

        blt::size_t get_closest_neuron(const std::vector<Scalar>& data);

        Scalar find_closest_neighbour_distance(blt::size_t v0);

        void train_epoch(Scalar initial_learn_rate);

        blt::vec2 get_topological_position(const std::vector<Scalar>& data);

        Scalar topological_error();

        Scalar quantization_error();

        void compute_errors();

        void compute_neuron_activations(Scalar distance = 2, Scalar activation = 0.5);

        void write_activations(std::ostream& out);

        void write_topology_errors(std::ostream& out);

        void write_quantization_errors(std::ostream& out);

        void write_all_errors(std::ostream& out);

        [[nodiscard]] const array_t& get_array() const
        {
            return array;
        }

        [[nodiscard]] blt::size_t get_current_epoch() const
        {
            return current_epoch;
        }

        [[nodiscard]] blt::size_t get_max_epochs() const
        {
            return max_epochs;
        }

        [[nodiscard]] const std::vector<Scalar>& get_topological_errors() const
        {
            return topological_errors;
        }

        [[nodiscard]] const std::vector<Scalar>& get_quantization_errors() const
        {
            return quantization_errors;
        }

    private:
        array_t array;
        data_file_t file;
        blt::size_t current_epoch = 0;
        blt::size_t max_epochs;
        distance_function_t* dist_func;
        topology_function_t* topology_function;

        // normalized value for which below this will be considered neural
        float quantization_distance = 0.25;

        std::vector<Scalar> topological_errors;
        std::vector<Scalar> quantization_errors;
    };
}

#endif //COSC_4P80_ASSIGNMENT_3_SOM_H
