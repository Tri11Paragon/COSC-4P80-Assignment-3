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
blt::size_t max_epochs = 100;
Scalar initial_learn_rate = 0.1;

void init(const blt::gfx::window_data&)
{
    using namespace blt::gfx;
    BLT_INFO("Hello World!");
    
    global_matrices.create_internals();
    resources.load_resources();
    renderer_2d.create();
    
    som = std::make_unique<som_t>(
            *std::find_if(files.begin(), files.end(), [](const data_file_t& v) { return v.data_points.begin()->bins.size() == 32; }),
            som_width, som_height, max_epochs);
}

void update(const blt::gfx::window_data& data)
{
    using namespace blt::gfx;
    global_matrices.update_perspectives(data.width, data.height, 90, 0.1, 2000);
    
    camera.update();
    camera.update_view(global_matrices);
    global_matrices.update();
    
    if (ImGui::Begin("Controls"))
    {
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
    
    auto& meow = *std::find_if(files.begin(), files.end(), [](const data_file_t& v) { return v.data_points.begin()->bins.size() == 32; });
    for (auto& v : meow.data_points)
    {
        auto nearest = som->get_closest_neuron(v.bins);
        activations[nearest] += v.is_bad ? -1 : 1;
    }
    
    blt::i64 max = *std::max_element(activations.begin(), activations.end());
    blt::i64 min = *std::min_element(activations.begin(), activations.end());
    
    for (auto [i, v] : blt::enumerate(som->get_array().get_map()))
    {
        float scale = 35;
        
        auto activation = activations[i];
        
        blt::vec4 color = blt::make_color(1,1,1);
        if (activation > 0)
            color = blt::make_color(0, static_cast<Scalar>(activation) / static_cast<Scalar>(max), 0);
        else if (activation < 0)
            color = blt::make_color(std::abs(static_cast<Scalar>(activation) / static_cast<Scalar>(min)), 0, 0);
        
        renderer_2d.drawPointInternal(color, point2d_t{v.get_x() * scale + scale, v.get_y() * scale + scale, scale});
    }
    
    renderer_2d.render(data.width, data.height);
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
