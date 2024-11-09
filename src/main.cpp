#include <iostream>
#include <blt/std/logging.h>
#include <blt/parse/argparse.h>
#include <blt/gfx/window.h>
#include "blt/gfx/renderer/resource_manager.h"
#include "blt/gfx/renderer/batch_2d_renderer.h"
#include "blt/gfx/renderer/camera.h"
#include <assign3/file.h>
#include <assign3/som.h>
#include <imgui.h>

using namespace assign3;

std::vector<data_file_t> files;
std::unique_ptr<som_t> som;

blt::gfx::matrix_state_manager global_matrices;
blt::gfx::resource_manager resources;
blt::gfx::batch_renderer_2d renderer_2d(resources, global_matrices);
blt::gfx::first_person_camera_2d camera;

blt::size_t som_width = 7;
blt::size_t som_height = 7;
blt::size_t max_epochs = 1000;
Scalar initial_learn_rate = 0.1;

int currently_selected_network = 0;
std::vector<std::string> map_files_names;

float neuron_scale = 35;
float draw_width = neuron_scale * static_cast<float>(som_width);
float draw_height = neuron_scale * static_cast<float>(som_height);

void generate_network(int selection)
{
    som = std::make_unique<som_t>(files[selection].normalize(), som_width, som_height, max_epochs);
}

const char* get_selection_string(void*, int selection)
{
    return map_files_names[selection].c_str();
}

void init(const blt::gfx::window_data&)
{
    using namespace blt::gfx;
    BLT_INFO("Hello World!");
    
    global_matrices.create_internals();
    resources.load_resources();
    renderer_2d.create();
    
    for (const auto& data : files)
        map_files_names.emplace_back(std::to_string(data.data_points.begin()->bins.size()));
    
    generate_network(currently_selected_network);
}

void update(const blt::gfx::window_data& window_data)
{
    using namespace blt::gfx;
    global_matrices.update_perspectives(window_data.width, window_data.height, 90, 0.1, 2000);
    
    camera.update();
    camera.update_view(global_matrices);
    global_matrices.update();
    
    if (ImGui::Begin("Controls"))
    {
        ImGui::Text("Network Select");
        if (ImGui::ListBox("##Network Select", &currently_selected_network, get_selection_string, nullptr, static_cast<int>(map_files_names.size())))
            generate_network(currently_selected_network);
        
        if (ImGui::Button("Run Epoch"))
        {
            static gaussian_function_t func;
            som->train_epoch(initial_learn_rate, &func);
        }
        static bool run;
        ImGui::Checkbox("Run to completion", &run);
        if (run)
        {
            static gaussian_function_t func;
            if (som->get_current_epoch() < som->get_max_epochs())
                som->train_epoch(initial_learn_rate, &func);
        }
        ImGui::Text("Epoch %ld / %ld", som->get_current_epoch(), som->get_max_epochs());
    }
    ImGui::End();
    
    static std::vector<blt::i64> activations;
    
    activations.clear();
    activations.resize(som->get_array().get_map().size());
    
    auto current_data_file = files[currently_selected_network].normalize();
    for (auto& v : current_data_file.data_points)
    {
        auto nearest = som->get_closest_neuron(v.bins);
        activations[nearest] += v.is_bad ? -1 : 1;
    }
    
    blt::i64 max = *std::max_element(activations.begin(), activations.end());
    blt::i64 min = *std::min_element(activations.begin(), activations.end());
    
    for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
    {
        auto activation = activations[i];
        
        blt::vec4 color = blt::make_color(1, 1, 1);
        if (activation > 0)
            color = blt::make_color(0, static_cast<Scalar>(activation) / static_cast<Scalar>(max), 0);
        else if (activation < 0)
            color = blt::make_color(std::abs(static_cast<Scalar>(activation) / static_cast<Scalar>(min)), 0, 0);
        
        renderer_2d.drawPointInternal(color,
                                      point2d_t{v.get_x() * neuron_scale + neuron_scale, v.get_y() * neuron_scale + neuron_scale, neuron_scale});
    }
    
    static std::vector<float> closest_type;
    closest_type.clear();
    closest_type.resize(som->get_array().get_map().size());
    
    for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
    {
        Scalar lowest_distance = std::numeric_limits<Scalar>::max();
        bool is_bad = false;
        for (const auto& data : current_data_file.data_points)
        {
            auto dist = v.dist(data.bins);
            if (dist < lowest_distance)
            {
                lowest_distance = dist;
                is_bad = data.is_bad;
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
        
        renderer_2d.drawPointInternal(color,
                                      point2d_t{draw_width + neuron_scale * 2 + v.get_x() * neuron_scale + neuron_scale,
                                                v.get_y() * neuron_scale + neuron_scale, neuron_scale});
    }
    
    renderer_2d.render(window_data.width, window_data.height);
}

void destroy(const blt::gfx::window_data&)
{
    global_matrices.cleanup();
    resources.cleanup();
    renderer_2d.cleanup();
    blt::gfx::cleanup();
    BLT_INFO("Goodbye World!");
}

int main(int argc, const char** argv)
{
    blt::arg_parse parser{};
    
    parser.addArgument(blt::arg_builder{"--file", "-f"}.setDefault("../data").setHelp("Path to data files").build());
    
    auto args = parser.parse_args(argc, argv);
    
    files = assign3::data_file_t::load_data_files_from_path(args.get<std::string>("file"));
    
    blt::gfx::init(blt::gfx::window_data{"My Sexy Window", init, update, destroy}.setSyncInterval(1));
}
