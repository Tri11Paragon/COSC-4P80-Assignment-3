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
#include <assign3/manager.h>
#include <blt/gfx/window.h>
#include <imgui.h>
#include <implot.h>

const char* get_selection_string(void* user_data, int selection)
{
    return (*reinterpret_cast<std::vector<std::string>*>(user_data))[selection].c_str();
}

namespace assign3
{
    
    render_info_t render_info_t::fill_screen(renderer_t& renderer, float neuron_scale, float x_padding, float y_padding, float w_padding,
                                             float h_padding)
    {
        render_info_t info;
        info.set_neuron_scale(neuron_scale);
        info.set_base_pos({x_padding, y_padding});
        
        float screen_width = static_cast<float>(blt::gfx::getWindowWidth()) - x_padding - w_padding;
        float screen_height = static_cast<float>(blt::gfx::getWindowHeight()) - y_padding - h_padding;
        
        float neuron_width = static_cast<float>(renderer.som_width) * neuron_scale;
        float neuron_height = static_cast<float>(renderer.som_height) * neuron_scale;
        
        float remain_width = screen_width - neuron_width;
        float remain_height = screen_height - neuron_height;
        
        info.set_neuron_padding({remain_width / static_cast<float>(renderer.som_width), remain_height / static_cast<float>(renderer.som_height)});
        
        return info;
    }
    
    void motor_data_t::update()
    {
        for (const auto& data : files)
            map_files_names.emplace_back(std::to_string(data.data_points.begin()->bins.size()));
    }
    
    void renderer_t::create()
    {
        fr2d.create(250, 2048);
        br2d.create();
        
        generate_network(currently_selected_network);
        update_graphics();
    }
    
    void renderer_t::cleanup()
    {
        fr2d.cleanup();
        br2d.cleanup();
    }
    
    void renderer_t::draw_som(const std::function<blt::vec4(neuron_t&)>& color_func, bool debug)
    {
    
    }
    
    void renderer_t::render()
    {
        using namespace blt::gfx;
        
        if (ImGui::Begin("Controls"))
        {
            ImGui::Text("Network Select");
            if (ImGui::ListBox("##Network Select", &currently_selected_network, get_selection_string, &motor_data.map_files_names,
                               static_cast<int>(motor_data.map_files_names.size())))
                generate_network(currently_selected_network);
            
            if (ImGui::Button("Run Epoch"))
            {
                som->train_epoch(initial_learn_rate, topology_function.get());
            }
            static bool run;
            ImGui::Checkbox("Run to completion", &run);
            if (run)
            {
                if (som->get_current_epoch() < som->get_max_epochs())
                    som->train_epoch(initial_learn_rate, topology_function.get());
            }
            ImGui::Text("Epoch %ld / %ld", som->get_current_epoch(), som->get_max_epochs());
        }
        ImGui::End();
        
        static std::vector<blt::i64> activations;
        
        activations.clear();
        activations.resize(som->get_array().get_map().size());
        
        auto current_data_file = motor_data.files[currently_selected_network].normalize();
        for (auto& v : current_data_file.data_points)
        {
            const auto nearest = som->get_closest_neuron(v.bins);
            activations[nearest] += v.is_bad ? -1 : 1;
        }
        
        const blt::i64 max = *std::max_element(activations.begin(), activations.end());
        const blt::i64 min = *std::min_element(activations.begin(), activations.end());
        
        for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
        {
            const auto activation = activations[i];
            
            blt::vec4 color = blt::make_color(1, 1, 1);
            if (activation > 0)
                color = blt::make_color(0, static_cast<Scalar>(activation) / static_cast<Scalar>(max), 0);
            else if (activation < 0)
                color = blt::make_color(std::abs(static_cast<Scalar>(activation) / static_cast<Scalar>(min)), 0, 0);
            
            br2d.drawPointInternal(color, point2d_t{v.get_x() * neuron_scale + neuron_scale, v.get_y() * neuron_scale + neuron_scale, neuron_scale});
        }
        
        static std::vector<float> closest_type;
        closest_type.clear();
        closest_type.resize(som->get_array().get_map().size());
        
        for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
        {
            Scalar lowest_distance = std::numeric_limits<Scalar>::max();
            bool is_bad = false;
            for (const auto& [is_bins_bad, bins] : current_data_file.data_points)
            {
                if (const auto dist = v.dist(bins); dist < lowest_distance)
                {
                    lowest_distance = dist;
                    is_bad = is_bins_bad;
                }
            }
            //        BLT_TRACE(is_bad ? -lowest_distance : lowest_distance);
            closest_type[i] = is_bad ? -lowest_distance : lowest_distance;
        }
        
        auto min_dist = *std::min_element(closest_type.begin(), closest_type.end());
        auto max_dist = *std::max_element(closest_type.begin(), closest_type.end());
        
        for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
        {
            auto type = closest_type[i];
            
            blt::vec4 color = blt::make_color(1, 1, 1);
            if (type >= 0)
                color = blt::make_color(0, 1 - (type / max_dist) + 0.1f, 0);
            else if (type < 0)
                color = blt::make_color(1 - (type / min_dist) + 0.1f, 0, 0);
            
            br2d.drawPointInternal(color,
                                   point2d_t{
                                           draw_width + neuron_scale * 2 + v.get_x() * neuron_scale + neuron_scale,
                                           v.get_y() * neuron_scale + neuron_scale, neuron_scale
                                   });
        }
        
        closest_type.clear();
        closest_type.resize(som->get_array().get_map().size());
        for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
        {
            auto half = som->find_closest_neighbour_distance(i);
            auto scale = topology_function->scale(half * 0.5f, 0.5);
            for (const auto& data : current_data_file.data_points)
            {
                auto dist = v.dist(data.bins);
                auto ds = topology_function->call(dist, scale);
                //            BLT_TRACE("%f, %f, %f", ds, dist, scale);
                if (data.is_bad)
                    closest_type[i] -= ds;
                else
                    closest_type[i] += ds;
            }
        }
        auto min_act = *std::min_element(closest_type.begin(), closest_type.end());
        auto max_act = *std::max_element(closest_type.begin(), closest_type.end());
        for (auto& v : closest_type)
        {
            auto n = 2 * (v - min_act) / (max_act - min_act) - 1;
            v = n;
        }
        //    BLT_TRACE("Min %f Max %f", min_act, max_act);
        
        for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
        {
            auto type = closest_type[i];
            
            blt::vec4 color;
            if (type >= 0)
                color = blt::make_color(0, type, 0);
            else
                color = blt::make_color(-type, 0, 0);
            
            br2d.drawPointInternal(color,
                                   point2d_t{
                                           draw_width + neuron_scale * 2 + v.get_x() * neuron_scale + neuron_scale,
                                           draw_height + neuron_scale * 2 + v.get_y() * neuron_scale + neuron_scale, neuron_scale
                                   });
        }
        
        
        br2d.render(0,0);
        fr2d.render();
    }
    
    void renderer_t::update_graphics()
    {
        // find the min x / y for the currently drawn som as positions may depend on type.
        const auto x_comparator = [](const auto& a, const auto& b) {
            return a.get_x() < b.get_x();
        };
        const auto y_comparator = [](const auto& a, const auto& b) {
            return a.get_y() < b.get_y();
        };
        const auto& som_neurons = som->get_array().get_map();
        auto min_x = std::min_element(som_neurons.begin(), som_neurons.end(), x_comparator)->get_x();
        auto max_x = std::max_element(som_neurons.begin(), som_neurons.end(), x_comparator)->get_x();
        auto min_y = std::min_element(som_neurons.begin(), som_neurons.end(), y_comparator)->get_y();
        auto max_y = std::max_element(som_neurons.begin(), som_neurons.end(), y_comparator)->get_y();
        draw_width = (max_x - min_x) * neuron_scale;
        draw_height = (max_y - min_y) * neuron_scale;
        
        topology_function = std::make_unique<gaussian_function_t>();
    }
}