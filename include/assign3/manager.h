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
    
    struct render_info_t
    {
        blt::vec2 base_pos;
        blt::vec2 neuron_padding;
        float neuron_scale = 0;
        
        static render_info_t fill_screen(renderer_t& renderer, float neuron_scale = 35, float x_padding = 250, float y_padding = 50, float w_padding = 50, float h_padding = 50);
        
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
    
    
    class renderer_t
    {
            friend motor_data_t;
            friend render_info_t;
        public:
            explicit renderer_t(motor_data_t& data, blt::gfx::resource_manager& resources, blt::gfx::matrix_state_manager& state):
                    motor_data(data), br2d{resources, state}
            {}
            
            void create();
            
            void cleanup();
            
            void draw_som(const std::function<blt::vec4(neuron_t&)>& color_func, bool debug);
            
            void render();
            
            void update_graphics();
            
            void generate_network(int selection)
            {
                som = std::make_unique<som_t>(motor_data.files[selection].normalize(), som_width, som_height, max_epochs);
            }
        
        private:
            motor_data_t& motor_data;
            std::unique_ptr<som_t> som;
            std::unique_ptr<topology_function_t> topology_function;
            
            blt::gfx::font_renderer_t fr2d{};
            blt::gfx::batch_renderer_2d br2d;
            
            float draw_width = 0;
            float draw_height = 0;
            float neuron_scale = 35;
            
            blt::size_t som_width = 5;
            blt::size_t som_height = 5;
            blt::size_t max_epochs = 100;
            Scalar initial_learn_rate = 0.1;
            
            int currently_selected_network = 0;
    };
    
}

#endif //COSC_4P80_ASSIGNMENT_3_UI_H
