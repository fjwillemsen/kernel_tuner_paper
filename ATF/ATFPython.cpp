#define PARALLEL_SEARCH_SPACE_GENERATION

#include "ATF/atf.hpp"
#include <map>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace py::literals;

int add(int i, int j)
{
#include "ATFPython_searchspacespec.cpp"
    return i + j;
}

std::map<char, int> tune(py::dict py_tune_params, py::list py_restrictions)
{
    // // Step 0: Convert Python types to C++ types
    // std::list<atf::tp_t<std::string, std::set<std::string>, std::function<bool(std::string)>>> tune_params;
    // auto restrictions = py_restrictions.cast<std::vector<std::string>>();
    // // auto tune_params = py_tune_params.cast<std::map<std::string, <std::list<std::int>>>>();

    // for (std::pair<py::handle, py::handle> item : py_tune_params)
    // {
    //     auto param_name = item.first.cast<std::string>();
    //     auto values_vector = item.second.cast<std::vector<int>>();
    //     // auto param_values = std::initializer_list<int> i(values_vector.data(), values_vector.data() + values_vector.size());
    //     auto param_values = {1, 2};
    //     auto tuning_parameter = atf::tuning_parameter(param_name, param_values);
    //     tune_params.push_back(tuning_parameter);
    //     // std::cout << param_name << ": " << param_values[0];
    // }

    // Step 1: Generate the Search Space
    auto TP1 = atf::tuning_parameter("TP1", atf::interval<int>(0, 10));
    auto TP2 = atf::tuning_parameter("TP2", atf::interval<int>(0, 10));

    // Step 2: Define a Cost Function
    auto zero_cf = [&](atf::configuration &config) -> double
    {
        return 0.0;
    };

    // Step 3: Explore the Search Space
    auto tuning_result = atf::tuner().silent(true).tuning_parameters(TP1, TP2).search_technique(atf::exhaustive()).tune(zero_cf);

    // Step 4: Return the Result
    std::map<char, int> result;
    result['E'] = tuning_result.number_of_evaluated_configs();
    result['V'] = tuning_result.number_of_valid_configs();
    result['I'] = tuning_result.number_of_invalid_configs();
    return result;
}

PYBIND11_MODULE(ATFPython, m)
{
    m.doc() = "Auto-Tuning Framework (ATF) by Rasch et al. integration using pybind11"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers and returns the result.");
    m.def("tune", &tune, "A function to tune a kernel with ATF, returns a dictionary.");
}
