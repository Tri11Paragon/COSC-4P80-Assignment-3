#include <blt/std/logging.h>
#include <blt/parse/argparse.h>
#include <blt/gfx/window.h>
#include "blt/gfx/renderer/resource_manager.h"
#include "blt/gfx/renderer/camera.h"
#include "implot.h"
#include <assign3/file.h>
#include <assign3/manager.h>

using namespace assign3;

blt::gfx::matrix_state_manager global_matrices;
blt::gfx::resource_manager resources;
blt::gfx::first_person_camera_2d camera;
assign3::motor_data_t data{};
assign3::renderer_t renderer{data, resources, global_matrices};

void init(const blt::gfx::window_data&)
{
    using namespace blt::gfx;
    BLT_INFO("Hello World!");
    
    global_matrices.create_internals();
    resources.load_resources();
    renderer.create();
    
    ImPlot::CreateContext();
}

void update(const blt::gfx::window_data& window_data)
{
    using namespace blt::gfx;
    global_matrices.update_perspectives(window_data.width, window_data.height, 90, 0.1, 2000);
    
    camera.update();
    camera.update_view(global_matrices);
    global_matrices.update();
    
    renderer.render();
}

void destroy(const blt::gfx::window_data&)
{
    global_matrices.cleanup();
    resources.cleanup();
    renderer.cleanup();
    ImPlot::DestroyContext();
    blt::gfx::cleanup();
    BLT_INFO("Goodbye World!");
}

void load_data_files(const std::string& str)
{
    data.files = assign3::data_file_t::load_data_files_from_path(str);
    for (auto& v : data.files)
        v = v.normalize();
    data.update();
}

void action_start_graphics(const std::vector<std::string>& argv_vector)
{
    blt::arg_parse parser{};
    
    parser.addArgument(blt::arg_builder{"--file", "-f"}
                               .setDefault("../data")
                               .setHelp("Path to data files").build());
    
    auto args = parser.parse_args(argv_vector);
    
    load_data_files(args.get<std::string>("file"));
    
    blt::gfx::init(blt::gfx::window_data{"My Sexy Window", init, update, destroy}.setSyncInterval(1).setMaximized(true));
}

void action_test(const std::vector<std::string>& argv_vector)
{
    blt::arg_parse parser{};
    
    parser.addArgument(blt::arg_builder{"--file", "-f"}
                               .setDefault("../data")
                               .setHelp("Path to data files").build());
    
    auto args = parser.parse_args(argv_vector);
    
    load_data_files(args.get<std::string>("file"));
    
    
}

void action_convert(const std::vector<std::string>& argv_vector)
{

}

int main(int argc, const char** argv)
{
    std::vector<std::string> argv_vector;
    for (int i = 0; i < argc; i++)
        argv_vector.emplace_back(argv[i]);
    
    blt::arg_parse parser{};
    
    parser.addArgument(blt::arg_builder{"action"}
                               .setAction(blt::arg_action_t::SUBCOMMAND)
                               .setHelp("Action to run. Can be: [graphics, test, convert]").build());
    
    auto args = parser.parse_args(argv_vector);
    
    if (!args.contains("action"))
    {
        BLT_ERROR("Please provide an action");
        return 0;
    }
    
    argv_vector.erase(argv_vector.begin() + 1);
    
    auto action = blt::string::toLowerCase(args.get<std::string>("action"));
    if (action == "graphics")
        action_start_graphics(argv_vector);
    else if (action == "test")
        action_test(argv_vector);
    else if (action == "convert")
        action_convert(argv_vector);
    
}
