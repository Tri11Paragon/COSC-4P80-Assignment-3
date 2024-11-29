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
#include <cstdlib>
#include <utility>
#include <blt/fs/loader.h>

void plot_heatmap(const std::string& path, const std::string& activations_csv, const blt::size_t bin_size, const std::string& subtitle)
{
#ifdef __linux__
    auto pwd = std::filesystem::current_path().string();
    if (!blt::string::ends_with(pwd, '/'))
        pwd += '/';
    const std::string command = "cd '" + path + "' && python3 '" + pwd + "../plot_heatmap.py' '" + activations_csv + "' '" + std::to_string(bin_size)
        +
        "' '" + subtitle + "'";
    BLT_TRACE(command);
    std::system(command.c_str());
#endif
}

void plot_line_graph(const std::string& path, const std::string& topological_csv, const std::string& quantization_csv, blt::size_t bin_size,
                     const std::string& subtitle, const std::string& subtitle2)
{
#ifdef __linux__
    auto pwd = std::filesystem::current_path().string();
    if (!blt::string::ends_with(pwd, '/'))
        pwd += '/';
    const std::string command = "cd '" + path + "' && python3 '" + pwd + "../plot_line_graph.py' \"" + topological_csv + "\" \"" + quantization_csv +
        "\" " + std::to_string(bin_size) + " true \"" + subtitle + "\" \"" + subtitle2 + "\"";
    BLT_TRACE(command);
    std::system(command.c_str());
#endif
}

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
    constexpr float color = 0.15;
    // constexpr float color = 1;
    glClearColor(color, color, color, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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

void write_csv(const std::vector<Scalar>& vec, const std::string& path, const std::string& header = "")
{
    std::ofstream stream{path};
    stream << header << std::endl;
    for (const auto v : vec)
        stream << v << std::endl;
}

void load_csv(std::vector<Scalar>& vec, const std::string& path)
{
    auto lines = blt::fs::getLinesFromFile(path);
    for (const auto& [i, line] : blt::enumerate(lines).skip(1))
        vec.push_back(std::stof(line));
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
    {
    }

    gaussian_function_t topology_func{};
    std::vector<std::vector<Scalar>> topological_errors{};
    std::vector<std::vector<Scalar>> quantization_errors{};
    std::vector<std::vector<Scalar>> activations{};
};

struct sortable_data_t
{
    std::string_view path;
    Scalar value;
    blt::u32 rank;

    sortable_data_t(const std::string_view& path, Scalar value, blt::u32 rank): path(path), value(value), rank(rank)
    {
    }

    friend bool operator==(const sortable_data_t& lhs, const sortable_data_t& rhs)
    {
        return lhs.value == rhs.value;
    }

    friend bool operator!=(const sortable_data_t& lhs, const sortable_data_t& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator<(const sortable_data_t& lhs, const sortable_data_t& rhs)
    {
        return lhs.value < rhs.value;
    }

    friend bool operator<=(const sortable_data_t& lhs, const sortable_data_t& rhs)
    {
        return !(rhs < lhs);
    }

    friend bool operator>(const sortable_data_t& lhs, const sortable_data_t& rhs)
    {
        return rhs < lhs;
    }

    friend bool operator>=(const sortable_data_t& lhs, const sortable_data_t& rhs)
    {
        return !(lhs < rhs);
    }
};

std::string make_path(const task_t& task)
{
    std::stringstream paths;
    paths << "bins-" << task.file->data_points.begin()->bins.size() << "/";
    paths << task.width << "x" << task.height << '-' << task.max_epochs << '/';
    std::string shape_name = shape_names[static_cast<int>(task.shape)];
    std::string init_name = init_names[static_cast<int>(task.init)];
    blt::string::replaceAll(shape_name, " ", "-");
    blt::string::replaceAll(init_name, " ", "-");
    paths << shape_name << '/';
    paths << init_name << '-' << task.initial_learn_rate << '/';
    return paths.str();
}

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

    // tasks.emplace_back(&data.files.back(), 5, 5, 2000, shape_t::GRID, init_t::COMPLETELY_RANDOM, 1);
    // tasks.emplace_back(&data.files.back(), 5, 5, 2000, shape_t::GRID, init_t::RANDOM_DATA, 1);
    // tasks.emplace_back(&data.files.back(), 5, 5, 2000, shape_t::GRID, init_t::SAMPLED_DATA, 1);
    for (auto& file : data.files)
    {
        for (blt::u32 size = 5; size <= 7; size++)
        {
            for (int shape = 0; shape < 4; shape++)
            {
                for (int init = 0; init < 3; init++)
                {
                    tasks.emplace_back(&file, size, size, 2000, static_cast<shape_t>(shape), static_cast<init_t>(init), 1);
                }
            }
        }
    }


    static blt::size_t runs = 30;

    for (blt::size_t _ = 0; _ < std::thread::hardware_concurrency(); _++)
    {
        threads.emplace_back([&task_mutex, &tasks]()
        {
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

                bool do_run = false;
                if (do_run)
                {
                    for (blt::size_t run = 0; run < runs; run++)
                    {
                        gaussian_function_t func{};
                        auto dist = distance_function_t::from_shape(task.shape, task.width, task.height);
                        auto som = std::make_unique<som_t>(*task.file, task.width, task.height, task.max_epochs, dist.get(),
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
                }
                auto path = make_path(task);
                std::filesystem::create_directories(path);

                std::vector<Scalar> average_topological_errors;
                std::vector<Scalar> average_quantization_errors;
                std::vector<Scalar> average_activations;
                std::vector<Scalar> stddev_topological_errors;
                std::vector<Scalar> stddev_quantization_errors;
                std::vector<Scalar> min_topological_errors;
                std::vector<Scalar> min_quantization_errors;
                std::vector<Scalar> last_topological_errors;
                std::vector<Scalar> last_quantization_errors;

                if (do_run)
                {
                    average_topological_errors.resize(task.topological_errors.begin()->size());
                    average_quantization_errors.resize(task.quantization_errors.begin()->size());
                    average_activations.resize(task.activations.begin()->size());
                    stddev_topological_errors.resize(task.topological_errors.begin()->size());
                    stddev_quantization_errors.resize(task.quantization_errors.begin()->size());

                    min_topological_errors.resize(runs);
                    min_quantization_errors.resize(runs);
                    last_topological_errors.resize(runs);
                    last_quantization_errors.resize(runs);

                    for (auto [i, v] : blt::enumerate(task.topological_errors))
                    {
                        min_topological_errors[i] = *std::min_element(v.begin(), v.end());
                        last_topological_errors[i] = v.back();
                    }
                    for (auto [i, v] : blt::enumerate(task.quantization_errors))
                    {
                        min_quantization_errors[i] = *std::min_element(v.begin(), v.end());
                        last_quantization_errors[i] = v.back();
                    }

                    for (const auto& vec : task.topological_errors)
                        for (auto [index, v] : blt::enumerate(vec))
                            average_topological_errors[index] += v;
                    for (const auto& vec : task.quantization_errors)
                        for (auto [index, v] : blt::enumerate(vec))
                            average_quantization_errors[index] += v;
                    for (const auto& vec : task.activations)
                        for (auto [index, v] : blt::enumerate(vec))
                            average_activations[index] += v;

                    // calculate mean per point
                    for (auto& v : average_topological_errors)
                        v /= static_cast<Scalar>(runs);
                    for (auto& v : average_quantization_errors)
                        v /= static_cast<Scalar>(runs);

                    for (auto [i, mean] : blt::in_pairs(average_topological_errors, average_quantization_errors).enumerate())
                    {
                        auto [t_mean, q_mean] = mean;
                        float variance_t = 0;
                        float variance_q = 0;
                        for (const auto& vec : task.topological_errors)
                        {
                            auto d = vec[i] - t_mean;
                            variance_t += d * d;
                        }
                        for (const auto& vec : task.quantization_errors)
                        {
                            auto d = vec[i] - q_mean;
                            variance_q += d * d;
                        }
                        variance_t /= static_cast<Scalar>(runs);
                        variance_q /= static_cast<Scalar>(runs);
                        stddev_topological_errors[i] = std::sqrt(variance_t);
                        stddev_quantization_errors[i] = std::sqrt(variance_q);
                    }
                }
                else
                {
                    load_csv(average_topological_errors, path + "topological_avg.csv");
                    load_csv(average_quantization_errors, path + "quantization_avg.csv");
                    load_csv(average_activations, path + "activations_avg.csv");
                    load_csv(stddev_topological_errors, path + "topological_stddev.csv");
                    load_csv(stddev_quantization_errors, path + "quantization_stddev.csv");
                    load_csv(min_topological_errors, path + "min_topological.csv");
                    load_csv(min_quantization_errors, path + "min_quantization.csv");
                    load_csv(last_topological_errors, path + "last_topological.csv");
                    load_csv(last_quantization_errors, path + "last_quantization.csv");
                }

                Scalar avg_quantization_stddev = 0;
                Scalar avg_topological_stddev = 0;

                for (auto [q, t] : blt::in_pairs(stddev_quantization_errors, stddev_topological_errors))
                {
                    avg_quantization_stddev += q;
                    avg_topological_stddev += t;
                }

                avg_quantization_stddev /= static_cast<Scalar>(task.max_epochs);
                avg_topological_stddev /= static_cast<Scalar>(task.max_epochs);

                auto min_quant =
                    *std::min_element(average_quantization_errors.begin(), average_quantization_errors.end());
                auto max_quant =
                    *std::max_element(average_quantization_errors.begin(), average_quantization_errors.end());

                auto min_topo = *std::min_element(average_topological_errors.begin(), average_topological_errors.end());
                auto max_topo = *std::max_element(average_topological_errors.begin(), average_topological_errors.end());

                if (do_run)
                {
                    std::ofstream topological{path + "topological_avg.csv"};
                    std::ofstream quantization{path + "quantization_avg.csv"};
                    std::ofstream activations_avg{path + "activations_avg.csv"};
                    std::ofstream activations{path + "activations.csv"};
                    std::ofstream topological_stddev{path + "topological_stddev.csv"};
                    std::ofstream quantization_stddev{path + "quantization_stddev.csv"};

                    write_csv(min_topological_errors, path + "min_topological.csv");
                    write_csv(min_quantization_errors, path + "min_quantization.csv");
                    write_csv(last_topological_errors, path + "last_topological.csv");
                    write_csv(last_quantization_errors, path + "last_quantization.csv");

                    topological_stddev << "Average topological stddev: " << avg_topological_stddev << std::endl;
                    quantization_stddev << "Average quantization stddev: " << avg_quantization_stddev << std::endl;
                    // topological_stddev << "Stddev Over Epochs: " << std::endl;
                    // quantization_stddev << "Stddev Over Epochs: " << std::endl;

                    for (auto v : stddev_topological_errors)
                        topological_stddev << v << std::endl;
                    for (auto v : stddev_quantization_errors)
                        quantization_stddev << v << std::endl;

                    topological << "error\n";
                    quantization << "error\n";
                    for (auto [i, v] : blt::enumerate(average_topological_errors))
                    {
                        topological << v << '\n';
                    }
                    for (auto [i, v] : blt::enumerate(average_quantization_errors))
                    {
                        quantization << v << '\n';
                    }
                    for (auto [i, v] : blt::enumerate(average_activations))
                    {
                        activations_avg << v / static_cast<Scalar>(runs);
                        if (i % task.width == task.width - 1)
                            activations_avg << '\n';
                        else
                            activations_avg << ',';
                    }
                    for (auto [i, v] : blt::enumerate(task.activations.front()))
                    {
                        activations << v;
                        if (i % task.width == task.width - 1)
                            activations << '\n';
                        else
                            activations << ',';
                    }
                }

                std::string shape_name = shape_names[static_cast<int>(task.shape)];
                std::string init_name = init_names[static_cast<int>(task.init)];
                blt::string::replaceAll(shape_name, " ", "-");
                blt::string::replaceAll(init_name, " ", "-");

                plot_heatmap(path, "activations.csv", task.file->data_points.front().bins.size(),
                             std::to_string(task.width) + "x" + std::to_string(task.height) + " " += shape_name + ", " += init_name + ", " +
                             std::to_string(
                                 task.max_epochs) +
                             " Epochs");

                plot_line_graph(path, "topological_avg.csv", "quantization_avg.csv", task.file->data_points.front().bins.size(),
                                std::to_string(task.width) + "x" + std::to_string(task.height) + " " += shape_name + ", " += init_name + ", Min: " +
                                std::to_string(min_topo) + ", Max: " + std::to_string(max_topo) +
                                ", " + std::to_string(task.max_epochs) + " Epochs",
                                std::to_string(task.width) + "x" + std::to_string(task.height) + " " += shape_name + ", " += init_name + ", Min: " +
                                std::to_string(min_quant) + ", Max: " +
                                std::to_string(max_quant) + ", " + std::to_string(task.max_epochs) +
                                " Epochs");

                BLT_INFO("Task '%s' Complete", path.c_str());
            }
            while (true);
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

struct man_whitney_t
{
    Scalar u1 = 0, u2 = 0;
    Scalar U = 0, meanU = 0, sigmaU = 0;
    Scalar z = 0, r = 0;

    std::string name1, name2;
};

struct test_t
{
    std::vector<task_t> tasks;
    std::string path;

    test_t() = default;

    test_t(const std::vector<task_t>& tasks, std::string path)
        : tasks(tasks), path(std::move(path))
    {
    }
};

double cumulativeNormal(const double x)
{
    // two tailed
    return 0.5 * std::erfc(-x * M_SQRT1_2) + (1.0 - 0.5 * std::erfc(x * M_SQRT1_2));
}

man_whitney_t do_man_whitney(const std::string& pop1_path, const std::string& pop2_path, const std::vector<sortable_data_t>& pop1,
                             const std::vector<sortable_data_t>& pop2)
{
    std::vector<sortable_data_t> data;
    data.insert(data.end(), pop1.begin(), pop1.end());
    data.insert(data.end(), pop2.begin(), pop2.end());

    std::sort(data.begin(), data.end());

    Scalar T1 = 0, T2 = 0;
    const auto n1 = static_cast<Scalar>(pop1.size()), n2 = static_cast<Scalar>(pop2.size());

    blt::u32 rank = 1;
    for (auto it = data.begin(); it != data.end();)
    {
        const auto begin = it;
        blt::size_t total_count = 1;
        blt::u32 total_rank = rank++;
        while ((it + 1) != data.end() && *begin == *(it + 1))
        {
            ++total_count;
            total_rank += rank++;
            ++it;
        }
        ++it;
        for (auto it2 = begin; it2 != it; ++it2)
        {
            if (it2->path == pop1_path)
                T1 += static_cast<Scalar>(total_rank) / static_cast<Scalar>(total_count);
            else if (it2->path == pop2_path)
                T2 += static_cast<Scalar>(total_rank) / static_cast<Scalar>(total_count);
            else
                BLT_ABORT(("Impossible Path " + std::string(it2->path)).c_str());
        }
    }

    man_whitney_t man;
    man.u1 = n1 * n2 + ((n1 * (n1 + 1)) / 2) - T1;
    man.u2 = n1 * n2 + ((n2 * (n2 + 1)) / 2) - T2;
    man.U = std::min(man.u1, man.u2);
    man.meanU = (n1 * n2) / 2;
    man.sigmaU = std::sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);
    man.z = (man.U - man.meanU) / man.sigmaU;
    man.r = std::abs(man.z) / std::sqrt(n1 + n2);

    man.name1 = pop1_path;
    man.name2 = pop2_path;

    return man;
}

void action_convert(const std::vector<std::string>& argv_vector)
{
    blt::arg_parse parser{};
    parser.setHelpExtras("convert");

    parser.addArgument(blt::arg_builder{"--file", "-f"}
                       .setDefault("../data")
                       .setHelp("Path to data files").build());

    auto args = parser.parse_args(argv_vector);

    load_data_files(args.get<std::string>("file"));

    std::vector<std::thread> threads;
    std::vector<test_t> tasks;
    std::mutex task_mutex;

    // for (auto& file : data.files)
    // {
    //     for (blt::u32 i = 5; i <= 7; i++)
    //     {
    //         for (blt::i32 shape = 0; shape < 4; shape++)
    //         {
    //             auto shape_v = static_cast<shape_t>(shape);
    //             tasks.emplace_back(std::vector{
    //                                    task_t{&file, i, i, 2000, shape_v, init_t::COMPLETELY_RANDOM, 1.0},
    //                                    task_t{&file, i, i, 2000, shape_v, init_t::RANDOM_DATA, 1.0},
    //                                    task_t{&file, i, i, 2000, shape_v, init_t::SAMPLED_DATA, 1.0}
    //                                }, "UnUsed");
    //         }
    //     }
    // }

    for (auto& file : data.files)
    {
        tasks.emplace_back(std::vector{
                               task_t{&file, 5, 5, 2000, shape_t::GRID_WRAP, init_t::COMPLETELY_RANDOM, 1.0},
                               task_t{&file, 6, 6, 2000, shape_t::GRID_WRAP, init_t::COMPLETELY_RANDOM, 1.0},
                               task_t{&file, 7, 7, 2000, shape_t::GRID_WRAP, init_t::COMPLETELY_RANDOM, 1.0}
                           }, "UnUsed");
    }

    for (blt::size_t i = 0; i < std::thread::hardware_concurrency(); i++)
    {
        threads.emplace_back([&]()
        {
            while (true)
            {
                test_t t;
                {
                    std::unique_lock lock(task_mutex);
                    if (tasks.empty())
                        break;
                    t = tasks.back();
                    tasks.pop_back();
                }

                std::vector<std::string> paths;
                blt::hashmap_t<std::string, std::vector<sortable_data_t>> data;
                blt::hashmap_t<std::string, const task_t*> task_data;
                for (const auto& task : t.tasks)
                {
                    auto path = make_path(task) + "last_topological.csv";
                    paths.push_back(path);
                    auto lines = blt::fs::getLinesFromFile(path);
                    for (const auto& line : blt::iterate(lines).skip(1))
                        data[path].emplace_back(paths.back(), std::stof(line), 0);
                    task_data[path] = &task;
                }

                std::string same = task_data.begin()->first;
                for (const auto& task : task_data)
                {
                    for (auto [i, c] : blt::enumerate(task.first))
                    {
                        if (i < same.length() && same[i] != task.first[i])
                            same[i] = '%';
                    }
                }
                auto lines = blt::string::split_sv(same, '/');
                std::string filtered_path = "stats/";
                blt::size_t index = 0;
                for (const auto& [i, line] : blt::enumerate(lines))
                {
                    if (blt::string::contains(line, '%'))
                    {
                        index = i;
                        continue;
                    }
                    filtered_path += line;
                    filtered_path += '/';
                }
                auto bin_line = blt::string::split(lines[0], '-');
                auto bin_size = std::stoi(bin_line[1]);
                std::filesystem::create_directories(filtered_path);
                BLT_TRACE("Writing to path %s", filtered_path.c_str());

                std::vector<man_whitney_t> mans;
                for (auto [i, pair] : blt::iterate(data.begin(), data.end()).enumerate())
                {
                    for (const auto& [path2, vec2] : blt::iterate(data.begin(), data.end()).skip(i + 1))
                        mans.emplace_back(do_man_whitney(pair.first, path2, pair.second, vec2));
                }

                std::ofstream stats{filtered_path + "results_table.txt"};
                stats << "\\begin{figure}[h!]\n\t\\centering" << std::endl;
                stats << "\t\\makebox[\\textwidth]{\\begin{tabular}{cc}" << std::endl << "\t\t";

                for (auto [i, task] : blt::enumerate(t.tasks))
                {
                    if (i != 0)
                    {
                        if (i % 2 == 0)
                            stats << "\\\\" << std::endl << "\t\t";
                        else
                            stats << " & \n\t\t";
                    }
                    stats << "\\includegraphics[width=0.6\\textwidth]{" << make_path(task) + "errors-topological" << bin_size << "}";
                }
                stats << "\\\\" << std::endl;
                stats << "\t\\end{tabular}}" << std::endl;
                stats << "\t\\caption{}\n\t\\label{fig:}" << std::endl;
                stats << "\\end{figure}" << std::endl << std::endl;

                stats << "\\begin{figure}[h!]\n\t\\centering" << std::endl;
                stats << "\t\\makebox[\\textwidth]{\\begin{tabular}{cc}" << std::endl << "\t\t";

                for (auto [i, task] : blt::enumerate(t.tasks))
                {
                    if (i != 0)
                    {
                        if (i % 2 == 0)
                            stats << "\\\\" << std::endl << "\t\t";
                        else
                            stats << " & \n\t\t";
                    }
                    stats << "\\includegraphics[width=0.6\\textwidth]{" << make_path(task) + "errors-topological" << bin_size << "}";
                }
                stats << "\\\\" << std::endl << "\t\t";
                for (auto [i, task] : blt::enumerate(t.tasks))
                {
                    if (i != 0)
                    {
                        if (i % 2 == 0)
                            stats << "\\\\" << std::endl << "\t\t";
                        else
                            stats << " & \n\t\t";
                    }
                    stats << "\\includegraphics[width=0.6\\textwidth]{" << make_path(task) + "errors-quantization" << bin_size << "}";
                }
                stats << "\\\\" << std::endl;
                stats << "\t\\end{tabular}}" << std::endl;
                stats << "\t\\caption{}\n\t\\label{fig:}" << std::endl;
                stats << "\\end{figure}" << std::endl << std::endl;

                stats << "\\begin{table}[h!]\n\t\\centering" << std::endl;
                stats <<
                    "\t\\makebox[\\textwidth]{\\begin{tabular}{||m{0.3\\linewidth}|m{0.125\\linewidth}|m{0.2\\linewidth}|m{0.2\\linewidth}|m{0.15\\linewidth}||}"
                    << std::endl;
                stats << "\t\t\\hline" << std::endl;
                stats << "\t\tName & Z-Value & P-Value & Effect Size & Significant\\\\" << std::endl;
                stats << "\t\t\\hline" << std::endl;
                for (const auto& man : mans)
                {
                    auto lines1 = blt::string::split(man.name1, '/');
                    auto lines2 = blt::string::split(man.name2, '/');
                    const auto& name1 = lines1[index];
                    const auto& name2 = lines2[index];

                    auto effect = man.r < 0.3 ? "Small" : (man.r < 0.5 ? "Medium" : "Large");
                    constexpr Scalar acceptance_region = 1.96;
                    auto sig = (man.z < -acceptance_region || man.z > acceptance_region) ? "Yes" : "No";

                    BLT_TRACE("Z: %f P: %f", man.z, cumulativeNormal(man.z));
                    stats << "\t\t" << name1 << " \\newline " << name2 << " & " << man.z << " & " << cumulativeNormal(man.z) << " & " << man.r << " ("
                        << effect << ") & " << sig << "\\\\" << std::endl;
                    stats << "\t\t\\hline" << std::endl;
                }
                stats << "\t\\end{tabular}}" << std::endl;
                stats << "\t\\caption{}\n\t\\label{tbl:}" << std::endl;
                stats << "\\end{table}" << std::endl;
            }
        });
    }

    for (auto& thread : threads)
    {
        if (thread.joinable())
            thread.join();
    }
}

int main(int argc, const char** argv)
{
    std::vector<std::string> argv_vector;
    for (int i = 0; i < argc; i++)
        argv_vector.emplace_back(argv[i]);

#ifdef __EMSCRIPTEN__
    action_start_graphics(argv_vector);
    return 0;
#endif

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
