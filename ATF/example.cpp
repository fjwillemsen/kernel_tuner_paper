#include <pybind11/pybind11.h>
#include "ATF/atf.hpp"

int add(int i, int j)
{
    return i + j;
}

int tune()
{
    // Step 1: Generate the Search Space
    auto opt_level = atf::tuning_parameter("opt_level", {"-O0", "-O1", "-O2", "-O3"});
    auto align_functions = atf::tuning_parameter("align_functions", {"-falign-functions", "-fno-align-functions"});
    auto early_inlining_insns = atf::tuning_parameter("early_inlining_insns", atf::interval<int>(0, 1000));

    // Step 2: Implement a Cost Function
    auto run_command = "./costfunction_empty.sh";
    auto compile_script = "../raytracer/compile_raytracer.sh";

    auto generic_cf = atf::generic::cost_function(run_command); //.compile_script(compile_script);

    // Step 3: Explore the Search Space
    auto tuning_result = atf::tuner().tuning_parameters(opt_level, align_functions, early_inlining_insns).search_technique(atf::auc_bandit()).tune(generic_cf, atf::duration<std::chrono::minutes>(1));
}

PYBIND11_MODULE(example, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("tune", &tune, "A function to tune a kernel with ATF");
}
