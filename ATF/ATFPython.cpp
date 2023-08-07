#define PARALLEL_SEARCH_SPACE_GENERATION

#include "ATF/atf.hpp"
#include <map>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// namespace py = pybind11;
// using namespace py::literals;

int add(int i, int j)
{
    return i + j;
}

std::map<char, int> tune()
{
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
