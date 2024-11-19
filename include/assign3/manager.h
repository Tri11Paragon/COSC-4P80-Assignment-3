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

#ifndef COSC_4P80_ASSIGNMENT_3_UI_H
#define COSC_4P80_ASSIGNMENT_3_UI_H

#include <blt/meta/codegen.h>
#include "blt/gfx/renderer/batch_2d_renderer.h"
#include "blt/gfx/renderer/font_renderer.h"
#include <assign3/file.h>
#include <assign3/som.h>
#include <functional>

namespace assign3
{
    class renderer_t;
    
    struct neuron_render_info_t
    {
        blt::vec2 base_pos;
        blt::vec2 neuron_padding;
        float neuron_scale = 0;
        
        static neuron_render_info_t fill_screen(renderer_t& renderer, float neuron_scale = 35, float x_padding = 250, float y_padding = 50,
                                                float w_padding = 50, float h_padding = 50);
        
        BLT_LVALUE_SETTER(blt::vec2, base_pos);
        
        BLT_LVALUE_SETTER(blt::vec2, neuron_padding);
        
        BLT_PRVALUE_SETTER(float, neuron_scale);
    };
    
    class motor_data_t
    {
        public:
            std::vector<data_file_t> files;
            std::vector<std::string> map_files_names;
            
            void update();
    };
    
    struct render_data_t
    {
        blt::size_t index;
        const neuron_t& neuron;
        blt::vec2 neuron_scaled;
        blt::vec2 neuron_offset;
        blt::vec2 neuron_padded;
        
        render_data_t(size_t index, const neuron_t& neuron, const blt::vec2& neuronScaled, const blt::vec2& neuronOffset,
                      const blt::vec2& neuronPadded):
                index(index), neuron(neuron), neuron_scaled(neuronScaled), neuron_offset(neuronOffset), neuron_padded(neuronPadded)
        {}
    };
    
    class renderer_t
    {
            friend motor_data_t;
            friend neuron_render_info_t;
        public:
            explicit renderer_t(motor_data_t& data, blt::gfx::resource_manager& resources, blt::gfx::matrix_state_manager& state):
                    motor_data(data), br2d{resources, state}
            {}
            
            void create();
            
            void cleanup();
            
            void draw_som(neuron_render_info_t info, const std::function<blt::vec4(render_data_t)>& color_func);
            
            void draw_debug(const data_file_t& file);
            
            void render();
            
            void regenerate_network()
            {
                switch (static_cast<shape_t>(selected_som_mode))
                {
                    case shape_t::GRID:
                        distance_function = std::make_unique<euclidean_distance_function_t>();
                        break;
                    case shape_t::GRID_WRAP:
                        distance_function = std::make_unique<toroidal_euclidean_distance_function_t>(som_width, som_height);
                        break;
                    case shape_t::GRID_OFFSET:
                        distance_function = std::make_unique<axial_distance_function_t>();
                        break;
                    case shape_t::GRID_OFFSET_WRAP:
                        distance_function = std::make_unique<toroidal_axial_distance_function_t>(som_width, som_height);
                        break;
                }
                som = std::make_unique<som_t>(motor_data.files[currently_selected_network], som_width, som_height, max_epochs,
                                              distance_function.get(), topology_function.get(), static_cast<shape_t>(selected_som_mode),
                                              static_cast<init_t>(selected_init_type), normalize_init);
                som->compute_neuron_activations();
            }
        
        private:
            motor_data_t& motor_data;
            std::unique_ptr<som_t> som;
            std::unique_ptr<topology_function_t> topology_function;
            std::unique_ptr<distance_function_t> distance_function;
            
            blt::gfx::font_renderer_t fr2d{};
            blt::gfx::batch_renderer_2d br2d;
            
            blt::i32 som_width = 5;
            blt::i32 som_height = 5;
            blt::i32 max_epochs = 2000;
            Scalar initial_learn_rate = 1;
            
            int currently_selected_network = 0;
            int selected_som_mode = 0;
            int selected_init_type = 0;
            bool normalize_init = false;
            bool debug_mode = false;
            bool draw_colors = true;
            bool draw_data_lines = false;
            bool running = false;
            int debug_state = 0;
            int selected_data_point = 0;
            int selected_neuron = 0;
            
            float requested_activation = 0.5;
            float at_distance_measurement = 2;
    };
    
}

#endif //COSC_4P80_ASSIGNMENT_3_UI_H
