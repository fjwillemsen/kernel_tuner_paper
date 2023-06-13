from pysmt.shortcuts import (
    Solver,
    Symbol,
    And,
    Or,
    Not,
    Equals,
    EqualsOrIff,
    GT,
    GE,
    LE,
    LT,
    Plus,
    Minus,
    Times,
    Div,
    Pow,
    Int,
    Real,
    String,
    Bool,
    get_model,
)
from pysmt.oracles import get_logic
from pysmt.typing import INT, STRING, REAL, BOOL

from test_searchspace import generate_searchspace

import re
import ast

boolean_comparison_mapping = {
    "=": Equals,
    "<": LT,
    "<=": LE,
    ">=": GE,
    ">": GT,
    "&&": And,
    "||": Or,
}

operators_mapping = {"+": Plus, "-": Minus, "*": Times, "/": Div, "^": Pow}

constant_init_mapping = {
    "int": Int,
    "float": Real,
    "str": String,
    "bool": Bool,
}

type_mapping = {
    "int": INT,
    "float": REAL,
    "str": STRING,
    "bool": BOOL,
}


def parse_restrictions(restrictions: list, tune_params: dict, symbols: dict):
    """parses restrictions from a list of strings into PySMT compatible restrictions"""
    regex_match_variable = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"

    def replace_params(match_object):
        key = match_object.group(1)
        if key in tune_params:
            return 'params["' + key + '"]'
        else:
            return key

    # rewrite the restrictions so variables are singled out
    parsed = [re.sub(regex_match_variable, replace_params, res) for res in restrictions]
    # ensure no duplicates are in the list
    parsed = list(set(parsed))
    print(parsed)

    # compile each restriction by replacing parameters and operators with their PySMT equivalent
    compiled_restrictions = list()
    for parsed_restriction in parsed:
        words = parsed_restriction.split(" ")

        # make a forward pass over all the words to organize and substitute
        var_or_constant_backlog = list()
        operator_backlog = list()
        boolean_backlog = list()
        for word in words:
            if word.startswith("params["):
                varname = word.replace('params["', "").replace('"]', "")
                var_or_constant_backlog.append(symbols[varname])
            elif word in boolean_comparison_mapping:
                boolean_backlog.append(boolean_comparison_mapping[word])
            elif word in operators_mapping:
                operator_backlog.append(operators_mapping[word])
            else:
                # evaluate the constant to check if it is an integer, float, etc. If not, treat it as a string.
                try:
                    constant = ast.literal_eval(word)
                except ValueError:
                    constant = word
                # convert from Python type to PySMT equivalent
                type_instance = constant_init_mapping[type(constant).__name__]
                var_or_constant_backlog.append(type_instance(constant))

        # for each of the operators, instantiate them with variables or constants
        for operator in operator_backlog:
            # merges the first two symbols in the backlog into one
            var_or_constant_backlog.insert(
                0,
                operator(
                    var_or_constant_backlog.pop(0), var_or_constant_backlog.pop(0)
                ),
            )

        # for each of the booleans, instantiate them with variables or constants
        compiled = list()
        assert len(boolean_backlog) <= 1, "Max. one boolean operator per restriction."
        for boolean in boolean_backlog:
            compiled.append(
                boolean(var_or_constant_backlog.pop(0), var_or_constant_backlog.pop(0))
            )

        # add the restriction to the list of restrictions
        compiled_restrictions.append(compiled[0])

    return And(compiled_restrictions)


def all_smt(formula, keys) -> list:
    target_logic = get_logic(formula)
    partial_models = list()
    with Solver(logic=target_logic) as solver:
        solver.add_assertion(formula)
        while solver.solve():
            partial_model = [EqualsOrIff(k, solver.get_value(k)) for k in keys]
            solver.add_assertion(Not(And(partial_model)))
            partial_models.append(partial_model)
    return partial_models


# get the tunable parameters and restrictions
tune_params, restrictions = generate_searchspace()

# setup each tunable parameter
symbols = dict([(v, Symbol(v, REAL)) for v in tune_params.keys()])
# symbols = [Symbol(v, REAL) for v in tune_params.keys()]
print(f"{symbols=}")

# for each tunable parameter, set the list of allowed values
domains = list()
for tune_param_key, tune_param_values in tune_params.items():
    domain = Or(
        [Equals(symbols[tune_param_key], Real(float(val))) for val in tune_param_values]
    )
    domains.append(domain)
domains = And(domains)
print(domains)

# add the restrictions
problem = parse_restrictions(restrictions, tune_params, symbols)
print(problem)

# combine the domain and restrictions
formula = And(domains, problem)
print(formula)

# get all solutions
keys = list(symbols.values())
from time import perf_counter

start = perf_counter()
all_solutions = all_smt(formula, keys)
print(perf_counter() - start)

# # get the solutions
# model = get_model(formula)
# if model:
#     print(model)
# else:
#     print("No solution found")


# hello = [Symbol(s, INT) for s in "hello"]
# world = [Symbol(s, INT) for s in "world"]
# letters = set(hello + world)
# domains = And([And(GE(l, Int(1)), LT(l, Int(10))) for l in letters])

# sum_hello = Plus(hello)  # n-ary operators can take lists
# sum_world = Plus(world)  # as arguments
# problem = And(Equals(sum_hello, sum_world), Equals(sum_hello, Int(25)))
# formula = And(domains, problem)

# print("Serialization of the formula:")
# print(formula)

# model = get_model(formula)
# if model:
#     print(model)
# else:
#     print("No solution found")
