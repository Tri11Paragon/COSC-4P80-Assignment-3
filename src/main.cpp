#include <blt/std/logging.h>
#include <blt/parse/argparse.h>
#include <blt/gfx/window.h>
#include "blt/gfx/renderer/resource_manager.h"
#include "blt/gfx/renderer/camera.h"
#include "implot.h"
#include <assign3/file.h>
#include <assign3/manager.h>
#include <thread>
#include <mutex>
#include <fstream>
#include <filesystem>

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
    parser.setHelpExtras("graphics");
    
    parser.addArgument(blt::arg_builder{"--file", "-f"}
                               .setDefault("../data")
                               .setHelp("Path to data files").build());
    
    auto args = parser.parse_args(argv_vector);
    
    load_data_files(args.get<std::string>("file"));
    
    blt::gfx::init(blt::gfx::window_data{"My Sexy Window", init, update, destroy}.setSyncInterval(1).setMaximized(true));
}

struct task_t // NOLINT
{
    data_file_t* file;
    blt::u32 width, height;
    blt::size_t max_epochs;
    shape_t shape;
    init_t init;
    Scalar initial_learn_rate;
    
    task_t() = default; // NOLINT
    
    task_t(data_file_t* file, blt::u32 width, blt::u32 height, size_t maxEpochs, shape_t shape, init_t init, Scalar initial_learn_rate):
            file(file), width(width), height(height), max_epochs(maxEpochs), shape(shape), init(init), initial_learn_rate(initial_learn_rate)
    {}
    
    gaussian_function_t topology_func{};
    std::vector<std::vector<Scalar>> topological_errors{};
    std::vector<std::vector<Scalar>> quantization_errors{};
    std::vector<std::vector<Scalar>> activations{};
};

void action_test(const std::vector<std::string>& argv_vector)
{
    blt::arg_parse parser{};
    parser.setHelpExtras("test");
    
    parser.addArgument(blt::arg_builder{"--file", "-f"}
                               .setDefault("../data")
                               .setHelp("Path to data files").build());
    
    auto args = parser.parse_args(argv_vector);
    
    load_data_files(args.get<std::string>("file"));
    
    std::vector<task_t> tasks;
    std::vector<std::thread> threads;
    std::mutex task_mutex;
    
    tasks.emplace_back(&data.files[1], 5, 5, 1000, shape_t::GRID, init_t::RANDOM_DATA, 0.1);
    
    static blt::size_t runs = 30;
    
    for (blt::size_t _ = 0; _ < std::thread::hardware_concurrency(); _++)
    {
        threads.emplace_back([&task_mutex, &tasks]() {
            do
            {
                task_t task;
                {
                    std::scoped_lock lock(task_mutex);
                    if (tasks.empty())
                        break;
                    task = std::move(tasks.back());
                    tasks.pop_back();
                }
                
                for (blt::size_t run = 0; run < runs; run++)
                {
                    gaussian_function_t func{};
                    auto dist = distance_function_t::from_shape(task.shape, task.width, task.height);
                    std::unique_ptr<som_t> som = std::make_unique<som_t>(*task.file, task.width, task.height, task.max_epochs, dist.get(),
                                                                         &task.topology_func, task.shape, task.init, false);
                    while (som->get_current_epoch() < som->get_max_epochs())
                        som->train_epoch(task.initial_learn_rate);
                    
                    task.topological_errors.push_back(som->get_topological_errors());
                    task.quantization_errors.push_back(som->get_quantization_errors());
                    
                    std::vector<Scalar> acts;
                    for (const auto& v : som->get_array().get_map())
                        acts.push_back(v.get_activation());
                    task.activations.emplace_back(std::move(acts));
                }
                std::stringstream paths;
                paths << "./bins-" << task.file->data_points.begin()->bins.size() << "/";
                paths << task.width << "x" << task.height << '-' << task.max_epochs << '/';
                std::string shape_name = shape_names[static_cast<int>(task.shape)];
                blt::string::replaceAll(shape_name, " ", "-");
                paths << shape_name << '/';
                std::string init_name = init_names[static_cast<int>(task.init)];
                blt::string::replaceAll(init_name, " ", "-");
                paths << init_name << '-' << task.initial_learn_rate << '/';
                auto path = paths.str();
                std::filesystem::create_directories(path);
                
                std::vector<Scalar> average_topological_errors;
                std::vector<Scalar> average_quantization_errors;
                std::vector<Scalar> average_activations;
                
                average_topological_errors.resize(task.topological_errors.begin()->size());
                average_quantization_errors.resize(task.quantization_errors.begin()->size());
                average_activations.resize(task.activations.begin()->size());
                
                for (const auto& vec : task.topological_errors)
                    for (auto [index, v] : blt::enumerate(vec))
                        average_topological_errors[index] += v;
                for (const auto& vec : task.quantization_errors)
                    for (auto [index, v] : blt::enumerate(vec))
                        average_quantization_errors[index] += v;
                for (const auto& vec : task.activations)
                    for (auto [index, v] : blt::enumerate(vec))
                        average_activations[index] += v;
                
                std::ofstream topological{path + "topological_avg.csv"};
                std::ofstream quantization{path + "quantization_avg.csv"};
                std::ofstream activations{path + "activations_avg.csv"};
                
                topological << "error\n";
                quantization << "error\n";
                for (auto [i, v] : blt::enumerate(average_topological_errors))
                {
                    topological << v / static_cast<Scalar>(runs) << '\n';
                }
                for (auto [i, v] : blt::enumerate(average_quantization_errors))
                {
                    quantization << v / static_cast<Scalar>(runs) << '\n';
                }
                for (auto [i, v] : blt::enumerate(average_activations))
                {
                    activations << v / static_cast<Scalar>(runs);
                    if (i % task.width == task.width-1)
                        activations << '\n';
                    else
                        activations << ',';
                }
                
                BLT_INFO("Task '%s' Complete", path.c_str());
                
            } while (true);
        });
    }
    
    while (!threads.empty())
    {
        if (threads.back().joinable())
        {
            threads.back().join();
            threads.pop_back();
        }
    }
}

void action_convert(const std::vector<std::string>& argv_vector)
{
    blt::arg_parse parser{};
    parser.setHelpExtras("convert");
    
    auto args = parser.parse_args(argv_vector);
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

//    argv_vector.erase(argv_vector.begin() + 1);
    
    auto action = blt::string::toLowerCase(args.get<std::string>("action"));
    if (action == "graphics")
        action_start_graphics(argv_vector);
    else if (action == "test")
        action_test(argv_vector);
    else if (action == "convert")
        action_convert(argv_vector);
    
}
