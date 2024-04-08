import pickle
import subprocess as sp
from pathlib import Path
from platform import python_version
from sys import platform
from typing import Any

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.util import default_block_size_names

default_max_threads = 1024


def ATF_specify_searchspace_in_source(tune_params: dict, restrictions: list, logfilename: str, path_prefix='ATF', sourcename='ATFPython_searchspacespec.cpp', block_size_names=default_block_size_names):
    """Replace the contents of the ATF source input file.

    Args:
        tune_params: dictionary of parameters to tune (keys) with their values.
        restrictions: the restrictions to apply on the searchspace.
        logfilename: the filename to write the ATF logs for, later used to construct the Searchspace.
        path_prefix: the path to the source. Defaults to 'ATF'.
        sourcename: the name of the source input file. Defaults to 'ATFPython_searchspacespec.cpp'.
    """
    def restriction_to_cpp(res: str) -> str:
        import re
        # regex_match_variable = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"
        # print(re.findall(regex_match_variable, res))

        # replace logic operators
        res = res.replace(' or ', ' || ').replace(' and ', ' && ').replace(' not ', ' != ')
        # replace consectutive comparisons (e.g. '1 < b <= 3' must become '(1 < b) && (b <= 3)')
        # first split into blocks of individual complete expressions
        subexpression_terminators = [' || ', ' && ']
        res_copy = res
        for terminator in subexpression_terminators:
            if terminator != subexpression_terminators[0]:
                res_copy = res_copy.replace(terminator, subexpression_terminators[0])
        # save the order of subexpression terminators so the full expression can be rebuilt later
        split_res_order = re.findall('|'.join(s.replace('|', '\|') for s in subexpression_terminators), res)
        split_res = res_copy.split(subexpression_terminators[0])
        # detect consecutive comparisons
        comparators = ['<=', '==', '!=', '>=', '>', '<']
        subexpressions = list()
        for r in split_res:
            comparator_indices = [(m.start(0), m.end(0)) for m in re.finditer('|'.join(comparators), r)]
            if len(comparator_indices) > 1:
                temps = list()
                for index in range(len(comparator_indices)):
                    temp_copy = r
                    prev_stop = comparator_indices[index-1][1] + 1 if index > 0 else 0
                    next_stop = comparator_indices[index+1][0] if index < len(comparator_indices) - 1 else len(temp_copy)
                    temp_copy = temp_copy[prev_stop:next_stop]
                    temps.append(f"({temp_copy.strip()})")
                subexpressions.append(f"({' && '.join(temps)})")
            else:
                subexpressions.append(f"({r})")
        res = ""
        for index, subexpression in enumerate(subexpressions):
            if index > 0:
                res += split_res_order[index-1]
            res += subexpression
        return res

    # get the relevant block size names and add the max_threads product restriction
    param_names = list(tune_params.keys())
    valid_block_size_names = list(block_size_name for block_size_name in block_size_names if block_size_name in param_names)
    if len(valid_block_size_names) > 0:
        max_threads = default_max_threads
        restrictions.append(f"{' * '.join(valid_block_size_names)} <= {max_threads}")

    # generate the restrictions specification (done before parameters because they must be added on the parameters)
    last_param_name = param_names[-1]
    restrictions_spec = f"[&](auto {last_param_name})" + "{ return ("
    restrictions_spec += ") && (".join(restriction_to_cpp(res) for res in restrictions)
    restrictions_spec += "); }"

    # generate the parameter specification
    # TODO implement case of independent restrictions
    parameters_spec = ""
    for param_name, values in tune_params.items():
        values_string = "{" + ', '.join(str(v) for v in values) + "}"
        # check if it can be an interval
        # intervals are not working (results in empty searchspace)
        if False and all(isinstance(v, int) for v in values) and len(values) > 2:
            min_v = min(values)
            max_v = max(values)
            if list(range(min_v, max_v+1)) == list(values):
                values_string = f"atf::interval<int>({min_v}, {max_v})"
            elif list(range(max_v, min_v-1, -1)) == list(values):
                values_string = f"atf::interval<int>({max_v}, {min_v})"
        parameters_spec += f'auto {param_name} = atf::tuning_parameter("{param_name}", {values_string}'
        if param_name == last_param_name:
            parameters_spec += f", {restrictions_spec}"
        parameters_spec += ");\n"

    # register the parameter names with ATF
    parameters_spec += "\n"
    param_names_spec = ", ".join(param_names)
    parameters_spec += f'auto tuner = atf::tuner().log_file("{path_prefix}/{logfilename}").silent(true).tuning_parameters({param_names_spec});'

    # put the generated specification in the source
    source = Path(path_prefix, sourcename)
    # assert source.exists() and source.is_file(), f"File {source} not found"
    source.unlink(missing_ok=True)
    source.touch()
    new = f"{parameters_spec}"
    source.write_text(new)
    assert source.exists() and source.is_file(), f"File {source} not found"

def ATF_compile(std='c++17', path_prefix='ATF'):
    """Compile the ATF source file.

    Args:
        std: the C++ standard to use. Defaults to 'c++17'.
        path_prefix: the path to the source. Defaults to 'ATF'.

    Raises:
        ValueError: in case of unsupported platform.
    """
    # set up environment specifics
    pyversion = '.'.join(python_version().split('.')[:-1]) # python version (major.minor)
    platform_specific = ""
    if platform == "linux":
        platform_specific = "-fPIC"
    elif platform == "darwin":
        platform_specific = "-undefined dynamic_lookup"
    else:
        raise ValueError(f"Platform {platform} not supported.")

    # resolve paths
    pybind11_path = Path(path_prefix, "extern/pybind11/include")
    source_path = Path(path_prefix, "ATFPython.cpp")
    assert pybind11_path.exists()
    assert source_path.exists()

    # define the full command
    command = f"c++ -O3 -shared -std={std} {platform_specific} $(python{pyversion}-config --includes) -I {pybind11_path} {source_path} -o {path_prefix}/ATFPython$(python{pyversion}-config --extension-suffix)"

    # compile by running the command
    sp.run(command, shell=True, text=True, check=True, capture_output=True)

def ATF_run() -> dict[str, Any]:
    """Function to Run ATF as a subprocess.

    Raises:
        CalledProcessError: in case the run fails.

    Returns:
        the results as a dictionary.
    """
    cmd = ['python', "./ATF/run_ATF.py"]
    # input_obj = pickle.dumps(tuple([tune_params, restrictions]))  # can be used with `input=input_obj` in sp.run
    try:
        result = sp.run(cmd, shell=False, capture_output=True, text=False, check=True)
    except sp.CalledProcessError as e:
        print(result.stderr)
        print(result.stdout)
        raise e
    return pickle.loads(result.stdout)

def ATF_result_searchspace(tune_params: dict, restrictions: list, logfilename: str, path_prefix='ATF') -> Searchspace:
    """Constructs a Searchspace object from the ATF logfile and returns it.

    Args:
        tune_params: dictionary of parameters to tune (keys) with their values.
        restrictions: the restrictions to apply on the searchspace.
        logfilename: the filename to write the ATF logs for, later used to construct the Searchspace.
        path_prefix: the path to the source. Defaults to 'ATF'.

    Returns:
        the Searchspace object.
    """
    # check whether there is a logfile
    path = Path(path_prefix, logfilename)
    assert path.exists()
    # construct the searchspace from the logfile
    ss = Searchspace(tune_params, restrictions, max_threads=default_max_threads, framework="ATF_cache", path_to_ATF_cache=path)
    # delete the logfile and return the searchspace object
    path.unlink()
    return ss
