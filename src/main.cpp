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

void init(const blt::gfx::window_data&)
{
    using namespace blt::gfx;
    BLT_INFO("Hello World!");
    
    global_matrices.create_internals();
    resources.load_resources();
    renderer_2d.create();
    
    blt::size_t size = 5;
    som = std::make_unique<som_t>(
            *std::find_if(files.begin(), files.end(), [](const data_file_t& v) { return v.data_points.begin()->bins.size() == 32; }),
            size, size, 100);
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
        ImGui::Button("Run Epoch");
        if (ImGui::IsItemClicked())
        {
            static gaussian_function_t func;
            som->train_epoch(0.1, &func);
        }
    }
    ImGui::End();
    
    auto& meow = *std::find_if(files.begin(), files.end(), [](const data_file_t& v) { return v.data_points.begin()->bins.size() == 32; });
    for (auto& v : som->get_array().get_map())
    {
        float scale = 35;
        
        float total_good_distance = 0;
        float total_bad_distance = 0;
        float total_goods = 0;
        float total_bads = 0;
        
        for (auto& point : meow.data_points)
        {
            auto dist = v.dist(point.bins);
            if (point.is_bad)
            {
                total_bads++;
                total_bad_distance += dist;
            } else
            {
                total_goods++;
                total_good_distance += dist;
            }
        }
        
        float good_ratio = total_goods > 0 ? total_good_distance / total_goods : 0;
        float bad_ratio = total_bads > 0 ? total_bad_distance / total_bads : 0;
        float good_to_bad = total_good_distance / total_bad_distance;
        
        BLT_TRACE("%f %f %f", good_ratio, bad_ratio, good_to_bad);
        
        renderer_2d.drawPointInternal(blt::make_color(good_ratio, bad_ratio, good_to_bad),
                                      point2d_t{v.get_x() * scale + scale, v.get_y() * scale + scale, scale});
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
