import gc

import torch
import pytest

from genparse import InferenceSetup
from lorem_ipsum import LOREM_IPSUM_GRAMMAR


# Lifted from benchmark_vllm_inference.py
RESTRICTED_SQL_GRAMMAR = r"""

start: WS? "SELECT" WS select_expr WS "FROM" WS from_expr [WS "WHERE" WS bool_condition] [WS "GROUP BY" WS var_list] [WS "ORDER BY" WS orderby_expr] WS EOS
EOS: "</s>"
select_expr: STAR | select_list
bool_condition: bool_expr | "(" bool_condition WS "AND" WS bool_condition ")" | "(" bool_condition WS "OR" WS bool_condition ")"
bool_expr: var "=" value | var ">" value | var "<" value
from_expr: "data"
orderby_expr: var_list WS "ASC" | var_list WS "DESC"
select_list: select_var ("," WS select_var)*
var_list: var ("," WS var)*
select_var: var | "AVG(" var ")" | "MEDIAN(" var ")" | "COUNT(" var ")"
var: "age" | "gender" | "year" | "state_color" | "zipcode" | "vote" | "race_ethnicity"
value: NUMBER | "'red'" | "'blue'" | "'white'" | "'black'" | "'latino'" | "'republican'" | "'democrat'" | "'male'" | "'female'"
STAR: "*"
NUMBER: /\d+/
WS: /[ \n\r\t]+/

"""
SQL_PROMPT = """You have access to a political survey data table named "data", which includes the following columns:
- "age" (integer)
- "gender" ("male" or "female"),
- "year" (integer)
- "state_color" ("blue" or "red")
- "zipcode" (integer)
- "vote" ("democrat" or "republican")
- "registered_party" ("democrat" or "republican")
- "race_ethnicity" ("white", "black", or "latino").

Q: Write a SQL query that shows individuals' age and gender, for people over 50 years old.
A: SELECT age, gender FROM data WHERE age>50 </s>
Q: Write a SQL query that shows individuals' vote and zipcode, ordered from lowest to highest age.
A: SELECT vote, zipcode, age FROM data ORDER BY age ASC </s>
Q: Write a SQL query that shows the old democrats in Williamsburg.
A:"""


def get_inference_setup(grammar: str):
    return InferenceSetup('gpt2', grammar, proposal_name='character')


# Reproduce the free_vllm_memory logic here so that we can run this benchmark with GPU on old
# commits for benchmark prototyping purposes.
def cleanup(inference_setup):
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )

        destroy_model_parallel()
        destroy_distributed_environment()

        try:
            del inference_setup.llm.llm_engine.model_executor
        except AttributeError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pass


def do_inference(inference_setup_: InferenceSetup, prompt: str, n_particles: int = 5):
    return inference_setup_(prompt, n_particles=n_particles, verbosity=1)


@pytest.mark.benchmark()
def test_very_long_sequences(benchmark):
    inference_setup = get_inference_setup(LOREM_IPSUM_GRAMMAR)
    benchmark(do_inference, inference_setup, 'Generate some lorem ipsum text:')
    cleanup(inference_setup)


@pytest.mark.benchmark()
def test_very_permissive_grammar(benchmark):
    inference_setup = get_inference_setup('start: /.+/')
    benchmark(do_inference, inference_setup, ' ')
    cleanup(inference_setup)


@pytest.mark.benchmark()
def test_many_particles(benchmark):
    inference_setup = get_inference_setup(RESTRICTED_SQL_GRAMMAR)
    benchmark(do_inference, inference_setup, SQL_PROMPT, n_particles=50)
    cleanup(inference_setup)
