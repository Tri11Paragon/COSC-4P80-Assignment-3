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
#include <blt/math/log_util.h>
#include <imgui.h>
#include <implot.h>

static void HelpMarker(const std::string& desc)
{
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip())
    {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

const char* get_selection_string(void* user_data, int selection)
{
    return (reinterpret_cast<std::string*>(user_data))[selection].c_str();
}

namespace assign3
{
    
    neuron_render_info_t neuron_render_info_t::fill_screen(renderer_t& renderer, float neuron_scale, float x_padding, float y_padding,
                                                           float w_padding,
                                                           float h_padding)
    {
        neuron_render_info_t info;
        info.set_neuron_scale(neuron_scale);
        info.set_base_pos({x_padding, y_padding});
        
        float screen_width = static_cast<float>(blt::gfx::getWindowWidth()) - x_padding - w_padding;
        float screen_height = static_cast<float>(blt::gfx::getWindowHeight()) - y_padding - h_padding;
        
        float neuron_width = static_cast<float>(renderer.som_width) * neuron_scale;
        float neuron_height = static_cast<float>(renderer.som_height) * neuron_scale;
        
        float remain_width = screen_width - neuron_width;
        float remain_height = screen_height - neuron_height;
        
        float remain = std::min(remain_width, remain_height);
        
        info.set_neuron_padding({remain / static_cast<float>(renderer.som_width), remain / static_cast<float>(renderer.som_height)});
        
        return info;
    }
    
    void motor_data_t::update()
    {
        for (const auto& data : files)
            map_files_names.emplace_back(std::to_string(data.data_points.begin()->bins.size()));
    }
    
    void renderer_t::create()
    {
        fr2d.create_default(250, 2048);
        br2d.create();
        
        topology_function = std::make_unique<gaussian_function_t>();
        
        regenerate_network();
        update_graphics();
    }
    
    void renderer_t::cleanup()
    {
        fr2d.cleanup();
        br2d.cleanup();
    }
    
    void renderer_t::draw_som(neuron_render_info_t info, const std::function<blt::vec4(render_data_t)>& color_func)
    {
        for (const auto& [i, neuron] : blt::enumerate(som->get_array().get_map()))
        {
            blt::vec2 neuron_pos = {neuron.get_x(), neuron.get_y()};
            auto neuron_scaled = neuron_pos * info.neuron_scale;
            auto neuron_offset = neuron_scaled + info.base_pos;
            auto neuron_padded = neuron_offset + neuron_pos * info.neuron_padding;
            
            auto color = color_func({i, neuron, neuron_scaled, neuron_offset, neuron_padded});
            br2d.drawPointInternal(color, blt::gfx::point2d_t{neuron_padded, info.neuron_scale});
        }
    }
    
    void renderer_t::render()
    {
        using namespace blt::gfx;
        
        ImGui::ShowDemoWindow();
        ImPlot::ShowDemoWindow();
        
        if (ImGui::Begin("Controls"))
        {
            ImGui::SetNextItemOpen(true, ImGuiCond_Appearing);
            if (ImGui::CollapsingHeader("SOM Control"))
            {
                ImGui::Text("Network Select");
                if (ImGui::ListBox("##Network Select", &currently_selected_network, get_selection_string, motor_data.map_files_names.data(),
                                   static_cast<int>(motor_data.map_files_names.size())))
                    regenerate_network();
                
                if (ImGui::Button("Run Epoch"))
                    som->train_epoch(initial_learn_rate, topology_function.get());
                ImGui::Checkbox("Run to completion", &running);
                ImGui::Text("Epoch %ld / %ld", som->get_current_epoch(), som->get_max_epochs());
            }
            if (ImGui::CollapsingHeader("SOM Settings"))
            {
                if (ImGui::InputInt("SOM Width", &som_width) || ImGui::InputInt("SOM Height", &som_height) ||
                    ImGui::InputInt("Max Epochs", &max_epochs))
                    regenerate_network();
                if (ImGui::InputFloat("Initial Learn Rate", &initial_learn_rate))
                    regenerate_network();
            }
            if (ImGui::CollapsingHeader("Debug"))
            {
                ImGui::Checkbox("Debug Visuals", &debug_mode);
                if (debug_mode)
                {
                    ImGui::ListBox("##DebugStateSelect", &debug_state, get_selection_string, debug_names.data(), debug_names.size());
                    switch (static_cast<debug_type>(debug_state))
                    {
                        case debug_type::DATA_POINT:
                        {
                            auto current_data_file = motor_data.files[currently_selected_network].normalize();
                            std::vector<std::string> names;
                            for (const auto& [i, v] : blt::enumerate(current_data_file.data_points))
                                names.push_back("#" + std::to_string(i) + " (" + (v.is_bad ? "Bad)" : "Good)"));
                            ImGui::Text("Select Data Point");
                            ImGui::ListBox("##SelectDataPoint", &selected_data_point, get_selection_string, names.data(),
                                           static_cast<int>(names.size()));
                        }
                            break;
                        case debug_type::DISTANCE:
                            
                            break;
                    }
                }
            }
        }
        ImGui::End();
        
        if (running)
        {
            if (som->get_current_epoch() < som->get_max_epochs())
                som->train_epoch(initial_learn_rate, topology_function.get());
        }
        
        auto current_data_file = motor_data.files[currently_selected_network].normalize();
        
        if (!debug_mode)
        {
            auto closest_type = get_neuron_activations(current_data_file);
            draw_som(neuron_render_info_t{}.set_base_pos({370, 145}).set_neuron_scale(120).set_neuron_padding({5, 5}),
                     [&closest_type](render_data_t context) {
                         auto type = closest_type[context.index];
                         return type >= 0 ? blt::make_color(0, type, 0) : blt::make_color(-type, 0, 0);
                     });
        } else
            draw_debug(current_data_file);
        
        br2d.render(0, 0);
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
        draw_width = (max_x - min_x);
        draw_height = (max_y - min_y);
    }
    
    std::vector<float> renderer_t::get_neuron_activations(const data_file_t& file)
    {
        static std::vector<float> closest_type;
        closest_type.clear();
        closest_type.resize(som->get_array().get_map().size());
        
        for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
        {
            auto half = som->find_closest_neighbour_distance(i);
            auto scale = topology_function->scale(half * 0.5f, 0.5);
            for (const auto& data : file.data_points)
            {
                auto dist = v.dist(data.bins);
                auto ds = topology_function->call(dist, scale);
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
        
        return closest_type;
    }
    
    void renderer_t::draw_debug(const data_file_t& file)
    {
        switch (static_cast<debug_type>(debug_state))
        {
            case debug_type::DATA_POINT:
            {
                std::vector<blt::vec2> data_positions;
                for (const auto& [i, v] : blt::enumerate(file.data_points))
                {
                    auto pos = som->get_topological_position(v.bins) * 120 + blt::vec2{370, 145};
                    auto color = blt::make_color(1,1,1);
                    float z_index = 1;
                    if (i == static_cast<blt::size_t>(selected_data_point))
                    {
                        color = blt::make_color(1, 0, 1);
                        z_index = 2;
                    }
                    br2d.drawRectangleInternal(color, blt::gfx::rectangle2d_t{pos, blt::vec2{8,8}}, z_index);
                }
                
                const auto& data_point = file.data_points[selected_data_point];
                auto closest_type = get_neuron_activations(file);
                draw_som(neuron_render_info_t{}.set_base_pos({370, 145}).set_neuron_scale(120).set_neuron_padding({0, 0}),
                         [this, &data_point, &closest_type](render_data_t context) {
                             auto& text = fr2d.render_text(std::to_string(context.neuron.dist(data_point.bins)), 18).setColor(0.2, 0.2, 0.8);
                             auto text_width = text.getAssociatedText().getTextWidth();
                             auto text_height = text.getAssociatedText().getTextHeight();
                             text.setPosition(context.neuron_padded - blt::vec2{text_width / 2.0f, text_height / 2.0f}).setZIndex(1);
                             
                             auto type = closest_type[context.index];
                             return type >= 0 ? blt::make_color(0, type, 0) : blt::make_color(-type, 0, 0);
                         });
            }
                break;
            case debug_type::DISTANCE:
            {
            
            }
                break;
        }
    }
}