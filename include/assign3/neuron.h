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

#ifndef COSC_4P80_ASSIGNMENT_3_NEURON_H
#define COSC_4P80_ASSIGNMENT_3_NEURON_H

#include <vector>
#include <assign3/fwdecl.h>
#include "blt/std/types.h"
#include <assign3/functions.h>
#include <assign3/file.h>

namespace assign3
{
    class neuron_t
    {
    public:
        explicit neuron_t(blt::size_t dimensions, Scalar x, Scalar y): x_pos(x), y_pos(y)
        {
            data.resize(dimensions);
        }

        neuron_t(const neuron_t&) = delete;
        neuron_t& operator=(const neuron_t&) = delete;

        neuron_t(neuron_t&&) = default;
        neuron_t& operator=(neuron_t&&) = default;

        neuron_t& randomize(blt::size_t seed, init_t init, bool normalize, const data_file_t& file);

        neuron_t& update(const std::vector<Scalar>& new_data, Scalar dist, Scalar eta);

        static Scalar distance(distance_function_t* dist_func, const neuron_t& n1, const neuron_t& n2);

        [[nodiscard]] Scalar dist(const std::vector<Scalar>& X) const;

        neuron_t& set_activation(Scalar act)
        {
            activation = act;
            return *this;
        }

        void activate(Scalar act)
        {
            activation += act;
        }

        [[nodiscard]] const std::vector<Scalar>& get_data() const
        {
            return data;
        }

        [[nodiscard]] Scalar get_x() const
        {
            return x_pos;
        }

        [[nodiscard]] Scalar get_y() const
        {
            return y_pos;
        }

        [[nodiscard]] Scalar get_activation() const
        {
            return activation;
        }

    private:
        Scalar x_pos, y_pos;
        Scalar activation = 0;
        std::vector<Scalar> data;
    };
}

#endif //COSC_4P80_ASSIGNMENT_3_NEURON_H
