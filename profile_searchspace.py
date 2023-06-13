"""A script to profile Searchspace initialization times."""

from test_searchspace import generate_searchspace, run_searchspace_initialization
import cProfile, pstats, io
import yappi

tune_params, restrictions = generate_searchspace()


def run():
    run_searchspace_initialization(tune_params=tune_params, restrictions=restrictions)


def profile_cprof():
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    pr.dump_stats("profile.prof")


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
