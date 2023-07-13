"""A script to profile Searchspace initialization times."""

import cProfile

import yappi
from constraint import check_if_compiled

from searchspaces_provider import dedispersion, expdist, generate_searchspace, hotspot
from test_searchspace import (
    assert_searchspace_validity,
    bruteforce_searchspace,
    run_searchspace_initialization,
)

tune_params, restrictions = generate_searchspace(cartesian_size=100000)
tune_params, restrictions, _, _, _, _ = expdist(restrictions_type="strings")
tune_params, restrictions, _, _, _, _ = dedispersion()
tune_params, restrictions, _, true_cartesian_size, _, _ = hotspot()


def run(check = True):
    print("python-constraint compiled") if check_if_compiled() else print("python-constraint not compiled")

    from time import perf_counter
    start = perf_counter()
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
