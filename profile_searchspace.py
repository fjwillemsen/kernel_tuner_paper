"""A script to profile Searchspace initialization times."""

import cProfile

import yappi

from searchspaces_provider import dedispersion, expdist, generate_searchspace, hotspot
from test_searchspace import (
    assert_searchspace_validity,
    bruteforce_searchspace,
    restrictions_strings_to_function,
    run_searchspace_initialization,
)

tune_params, restrictions = generate_searchspace(cartesian_size=100000)
tune_params, restrictions, _, _, _, ssname = expdist()
tune_params, restrictions, _, _, _, ssname = dedispersion()
tune_params, restrictions, _, _, _, ssname = hotspot()

if ssname:
    print(f"Profiling for searchspace {ssname}")


def run(check = False):
    try:
        from constraint import check_if_compiled
        print("python-constraint compiled") if check_if_compiled() else print("python-constraint not compiled")
        installed_unoptimized = False
    except ImportError:
        print("python-constraint-old")
        installed_unoptimized = True

    from time import perf_counter
    start = perf_counter()
    if installed_unoptimized:
        ss = run_searchspace_initialization(tune_params=tune_params, restrictions=restrictions_strings_to_function(restrictions, tune_params), framework='PythonConstraint')
    else:
        ss = run_searchspace_initialization(tune_params=tune_params, restrictions=restrictions, framework='PythonConstraint', kwargs=dict({'solver_method': 'PC_OptimizedBacktrackingSolver'}))
    print(f"Total time: {round(perf_counter() - start, 5)} seconds")
    if check:
        start = perf_counter()
        bruteforced = bruteforce_searchspace(tune_params, restrictions)
        print(f"Total time bruteforce: {round(perf_counter() - start, 5)} seconds")
        assert_searchspace_validity(bruteforced, ss)


def profile_cprof():
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    pr.dump_stats("profile.prof")
    pr.print_stats()


def profile_yappi():
    """Entry point for execution."""
    yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
    yappi.start()
    run()
    yappi.stop()
    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()


if __name__ == "__main__":
    run()
    # profile_cprof()
    # profile_yappi()
