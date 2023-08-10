#define PARALLEL_SEARCH_SPACE_GENERATION

#include "ATF/atf.hpp"
#include <map>
#include <string>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace py::literals;

int add(int i, int j)
{
    return i + j;
}

std::map<char, long int> tune()
{
    typedef std::chrono::high_resolution_clock Clock;
    auto start_time = Clock::now();

    // Step 1: Generate the Search Space Parameters & Constraints
#include "ATFPython_searchspacespec.cpp"

    // Step 2: Define a Cost Function
    auto zero_cf = [&](atf::configuration &config) -> double
    {
        return 0.0;
    };

    // Step 3: Explore the Search Space
    auto tuning_result = tuner.search_technique(atf::exhaustive()).tune(zero_cf);
    long int time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count(); // time taken in nanoseconds

    // Step 4: Return the Result
    std::map<char, long int> result;
    result['E'] = tuning_result.number_of_evaluated_configs();
    result['V'] = tuning_result.number_of_valid_configs();
    result['I'] = tuning_result.number_of_invalid_configs();
    result['T'] = time_taken;
    return result;
}

PYBIND11_MODULE(ATFPython, m)
{
    m.doc() = "Auto-Tuning Framework (ATF) by Rasch et al. integration using pybind11"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers and returns the result.");
    m.def("tune", &tune, "A function to tune a kernel with ATF, returns a dictionary containing the results.");
}
