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
#include <algorithm>

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

std::vector<assign3::Scalar> rotate90Clockwise(const std::vector<assign3::Scalar>& input, int width, int height)
{
    std::vector<assign3::Scalar> rotated(width * height);
    
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            int newRow = col;
            int newCol = height - row - 1;
            rotated[newRow * height + newCol] = input[row * width + col];
        }
    }
    
    return rotated;
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
        
//        ImGui::ShowDemoWindow();
//        ImPlot::ShowDemoWindow();
        
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
                {
                    som->train_epoch(initial_learn_rate);
                    som->compute_neuron_activations();
                }
                ImGui::Checkbox("Run to completion", &running);
                ImGui::Text("Epoch %ld / %ld", som->get_current_epoch(), som->get_max_epochs());
            }
            ImGui::SetNextItemOpen(true, ImGuiCond_Appearing);
            if (ImGui::CollapsingHeader("SOM Settings"))
            {
                ImGui::SeparatorText("Network Shape");
                if (ImGui::ListBox("##NetworkShape", &selected_som_mode, get_selection_string, shape_names.data(),
                                   static_cast<int>(shape_names.size())))
                    regenerate_network();
                ImGui::SeparatorText("Init Type");
                if (ImGui::ListBox("##InitType", &selected_init_type, get_selection_string, init_names.data(), static_cast<int>(init_names.size())))
                    regenerate_network();
                ImGui::TextWrapped("Help: %s", init_helps[selected_init_type].c_str());
                if (ImGui::Checkbox("Normalize Init Data", &normalize_init))
                    regenerate_network();
                ImGui::SeparatorText("Som Specifics");
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
                    switch (static_cast<debug_t>(debug_state))
                    {
                        case debug_t::DATA_POINT:
                        {
                            ImGui::Checkbox("Data Type Color", &draw_colors);
                            ImGui::Checkbox("Data Lines", &draw_data_lines);
                            auto current_data_file = motor_data.files[currently_selected_network];
                            static std::vector<std::string> names;
                            names.clear();
                            for (const auto& [i, v] : blt::enumerate(current_data_file.data_points))
                                names.push_back("#" + std::to_string(i) + " (" + (v.is_bad ? "Bad)" : "Good)"));
                            ImGui::Text("Select Data Point");
                            ImGui::ListBox("##SelectDataPoint", &selected_data_point, get_selection_string, names.data(),
                                           static_cast<int>(names.size()));
                        }
                            break;
                        case debug_t::DISTANCE:
                        {
                            static std::vector<std::string> names;
                            names.clear();
                            for (blt::size_t i = 0; i < som->get_array().get_map().size(); i++)
                            {
                                auto pos = som->get_array().from_index(i);
                                names.push_back("Neuron " + std::to_string(i) +
                                                " (" + std::to_string(pos.x()) + ", " + std::to_string(pos.y()) + ")");
                            }
                            ImGui::Text("Select Neuron");
                            ImGui::ListBox("##SelectNeuron", &selected_neuron, get_selection_string, names.data(), static_cast<int>(names.size()));
                        }
                            break;
                    }
                }
            }
        }
        ImGui::End();
        
        auto current_data_file = motor_data.files[currently_selected_network];
        
        if (ImGui::Begin("Plots & Data"))
        {
            ImPlot::SetNextAxesLimits(0, som_width, 0, som_height, ImPlotCond_Always);
            if (ImPlot::BeginPlot("Activations", ImVec2(-1, 0), ImPlotFlags_NoInputs))
            {
                static std::vector<float> activations;
                activations.clear();
                for (const auto& n : som->get_array().get_map())
                    activations.push_back(n.get_activation());
                auto rev = rotate90Clockwise(activations, som_width, som_height);
//                auto rev = closest_type;
//                std::reverse(rev.begin(), rev.end());
                ImPlot::PlotHeatmap("##data_map", rev.data(), som_height, som_width, 0, 0, "%.1f", ImPlotPoint(0, 0),
                                    ImPlotPoint(som_width, som_height), ImPlotHeatmapFlags_ColMajor);
                ImPlot::EndPlot();
            }
            ImPlot::SetNextAxesLimits(0, max_epochs, 0, 1, ImPlotCond_Once);
            if (ImPlot::BeginPlot("TError"))
            {
                ImPlot::PlotLine("##Terror", som->get_topological_errors().data(), static_cast<int>(som->get_topological_errors().size()));
                ImPlot::EndPlot();
            }
            ImPlot::SetNextAxesLimits(0, max_epochs, 0, static_cast<int>(current_data_file.data_points.size()), ImPlotCond_Once);
            if (ImPlot::BeginPlot("QError"))
            {
                ImPlot::PlotLine("##Qerror", som->get_quantization_errors().data(), static_cast<int>(som->get_quantization_errors().size()));
                ImPlot::EndPlot();
            }
        }
        ImGui::End();
        
        if (running)
        {
            if (som->get_current_epoch() < som->get_max_epochs())
                som->train_epoch(initial_learn_rate);
        }
        
        
        if (!debug_mode)
        {
            draw_som(neuron_render_info_t{}.set_base_pos({370, 145}).set_neuron_scale(120).set_neuron_padding({5, 5}),
                     [](render_data_t context) {
                         auto type = context.neuron.get_activation();
                         return type >= 0 ? blt::make_color(0, type, 0) : blt::make_color(-type, 0, 0);
                     });
        } else
            draw_debug(current_data_file);
        
        br2d.render(0, 0);
        fr2d.render();
    }
    
    void renderer_t::draw_debug(const data_file_t& file)
    {
        switch (static_cast<debug_t>(debug_state))
        {
            case debug_t::DATA_POINT:
            {
                std::vector<blt::vec2> data_positions;
                std::vector<blt::vec2> neuron_positions;
                for (const auto& [i, v] : blt::enumerate(file.data_points))
                {
                    auto pos = som->get_topological_position(v.bins) * 120 + blt::vec2{370, 145};
                    data_positions.push_back(pos);
                    auto color = blt::make_color(1, 1, 1);
                    float z_index = 2;
                    if (draw_colors)
                    {
                        if (v.is_bad)
                            color = blt::make_color(1, 0, 0);
                        else
                            color = blt::make_color(0, 1, 0);
                    }
                    if (i == static_cast<blt::size_t>(selected_data_point))
                    {
                        if (!draw_colors)
                        {
                            color = blt::make_color(1, 0, 1);
                            z_index = 3;
                        }
                    }
                    br2d.drawRectangleInternal(color, blt::gfx::rectangle2d_t{pos, blt::vec2{8, 8}}, z_index);
                }
                
                const auto& data_point = file.data_points[selected_data_point];
                draw_som(neuron_render_info_t{}.set_base_pos({370, 145}).set_neuron_scale(120).set_neuron_padding({0, 0}),
                         [this, &neuron_positions, &data_point](render_data_t context) {
                             auto half = som->find_closest_neighbour_distance(context.index) / at_distance_measurement;
                             auto scale = topology_function->scale(half, requested_activation);
                             auto ds = topology_function->call(context.neuron.dist(data_point.bins), scale);
                             
                             if (draw_data_lines)
                                 neuron_positions.push_back(context.neuron_padded);
                             auto& text = fr2d.render_text(std::to_string(ds), 18).setColor(0.2, 0.2, 0.8);
                             auto text_width = text.getAssociatedText().getTextWidth();
                             auto text_height = text.getAssociatedText().getTextHeight();
                             text.setPosition(context.neuron_padded - blt::vec2{text_width / 2.0f, text_height / 2.0f}).setZIndex(1);
                             
                             auto type = context.neuron.get_activation();
                             return type >= 0 ? blt::make_color(0, type, 0) : blt::make_color(-type, 0, 0);
                         });
                
                if (draw_data_lines)
                {
                    for (const auto& neuron : neuron_positions)
                        br2d.drawLineInternal(blt::make_color(1, 1, 0), blt::gfx::line2d_t{neuron, data_positions[selected_data_point]}, 1);
                }
            }
                break;
            case debug_t::DISTANCE:
            {
                auto& selected_neuron_ref = som->get_array().get_map()[selected_neuron];
                static std::vector<Scalar> distances_2d;
                static std::vector<Scalar> distances_nd;
                distances_2d.clear();
                distances_nd.clear();
                
                for (const auto& n : som->get_array().get_map())
                {
                    distances_2d.push_back(neuron_t::distance(distance_function.get(), selected_neuron_ref, n));
                    distances_nd.push_back(selected_neuron_ref.dist(n.get_data()));
                }
                
                draw_som(neuron_render_info_t{}.set_base_pos({370, 145}).set_neuron_scale(120).set_neuron_padding({0, 0}),
                         [this](render_data_t context) {
                             auto& text = fr2d.render_text(
                                     "2D: " + std::to_string(distances_2d[context.index]) + "\nND: " +
                                     std::to_string(distances_nd[context.index]), 18).setColor(0.2, 0.2, 0.8);
                             auto text_width = text.getAssociatedText().getTextWidth();
                             text.setPosition(context.neuron_padded - blt::vec2{text_width / 2.0f, 0}).setZIndex(1);
                             
                             if (static_cast<blt::size_t>(selected_neuron) == context.index)
                                 return blt::make_color(0, 0, 1);
                             auto type = context.neuron.get_activation();
                             return type >= 0 ? blt::make_color(0, type, 0) : blt::make_color(-type, 0, 0);
                         });
            }
                break;
        }
    }
}