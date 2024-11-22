GenParse currently provides a high-level interface for constrained generation via the `InferenceSetup` class. We recommend using this class as its internals may be deprecated without prior warning. 

```python
from genparse import InferenceSetup
```

## 1. Define your grammar 

GenParse uses Lark syntax for grammar specification. For example:

```python
grammar = """
start: WS? "SELECT" WS column WS from_clause (WS group_clause)?
from_clause: "FROM" WS table
group_clause: "GROUP BY" WS column
column: "age" | "name"
table: "employees"
WS: " "
"""
```

For a comprehensive guide on how to write grammars using Lark syntax, please refer to the [official Lark documentation](https://lark-parser.readthedocs.io/en/latest/grammar.html).

> **💡Tip:** GenParse supports grammars with arbitrary regular expressions. In practice, we recommend avoiding extremely permisive regular expressions (e.g., `/.+/`) since these will lead to significantly slower inference. See [issue #62](https://github.com/probcomp/genparse/issues/62).

## 2. Create an `InferenceSetup` object

Create an `InferenceSetup` object with your chosen language model and grammar:

```python
setup = InferenceSetup('gpt2', grammar)
```

`InferenceSetup` requires the following arguments:

- **model_name** (str): Name of the language model to use. See the README for the list of models currently supported by GenParse.
- **grammar** (str): The grammar specification in Lark format.


It accepts the following optional arguments:

- **proposal_name** (str): The type of proposal to use. Options include 'character' and 'token'. Default is 'character'.
- **num_processes** (int): The number of processes to use for parallel proposals. This can help speed up the inference process by utilizing multiple CPU cores. Default: min(mp.cpu_count(), 2)
- **use_rust_parser** (bool): Whether to use the Rust implementation of the Earley parser for faster inference. If False, the Python implementation is used. Default to True.
- **use_vllm** (bool or None): Whether to use VLLM for LLM next token probability computations. If None, VLLM is used when possible (i.e., if the vllm library is available and CUDA is enabled). Default is None.
- **seed** (int or None): A random seed for reproducibility. If provided, it ensures that the inference process is deterministic.
- **guide_opts** (dict or None): Additional options for the guide, which may include specific configurations for the grammar-based model.
- **proposal_opts** (dict or None): Additional options for the proposal mechanism, such as parameters specific to the proposal type (e.g., K for token proposal).
- **llm_opts** (dict or None): Additional options for the language model, such as temperature or top-p settings for sampling.
- **vllm_engine_opts** (dict or None): Additional options for the VLLM engine, such as data type (dtype). These options are ignored if VLLM is not used.


> **💡Tip:** To try different grammars without having to instantiate new `InferenceSetup` objects each time, use the `update_grammar` method; `setup.update_grammar(new_grammar)` will replace the existing grammar in `setup` with `new_grammar`.


## 3. Run inference

Use the setup object to run SMC inference:

```python
# The result is a ParticleApproximation object
result = setup('Write an SQL query:', n_particles=10, verbosity=1, max_tokens=25)
```

When calling `InferenceSetup`, the following arguments are required:

- **prompt** (str): The input prompt to generate samples from. This is the starting text for the language model.
- **n_particles** (int): The number of particles (samples) to generate.

There are the following optional arguments:

- **method** (str): The sampling method to use. Options include 'smc' for Sequential Monte Carlo and 'is' for importance sampling. Default to 'smc'.
* **max_tokens** (int): The maximum number of tokens to generate. Defaults to 500.

The following optional arguments are passed in as **kwargs**, which may be expanded over time:

- **potential** (Callable): A function that when called on a list of particles, returns a list with the log potential values for each particle. Potentials can be used to guide generation with additional constraints. See below for an overview of potential functions.
- **ess_threshold** (float): Effective sample size below which resampling is triggered, given as a fraction of **n_particles**. Default is 0.5.
* **verbosity** (int): Verbosity level. When > 0, particles are printed at each step. Default is 0.
- **return_record** (bool): Flag indicating whether to return a record of the inference steps. Default is False.
- **resample_method** (str): Resampling method to use. Either 'multinomial' or 'stratified'. Default is 'multinomial'.

The result from `InferenceSetup` is a `ParticleApproximation` object. This object contains a collection of particles, each representing a generated text sequence. Each particle has two main attributes:

- `context`: The generated text sequence.
- `weight`: A numerical value representing the particle's importance weight. The weights are not normalized probabilities. GenParse provides post-processing to convert these weights into meaningful probabilities, which can be accessed via the `.posterior` property:
```python
>>> result.posterior
{"SELECT name FROM employees GROUP BY name▪" : 1}
```

## 4. Potential functions

> **💡 Tip:** Incorporate constraints directly into the grammar when possible, as this will generally improve the quality of inference.

Potential functions can be used to guide generation using additional constraints. A potential function maps (partial) generations to positive real numbers, with higher values indicating a stronger preference for those generations. Intuitively, when applied in SMC, potential functions offer richer signals for resampling steps, allowing computation to be redirected toward more promising particles during the course of generation.

Potentials are provided as input to an `InferenceSetup` call via the `potential` argument and must be defined at the particle beam level. That is, `InferenceSetup` expects potentials to be callables which are provided a *list* of particles as input and return a *list* of log potential values, one for each particle. 

## 5. Visualizing inference

GenParse additionally provides methods to visualize inference runs. To display the visualization of an inference run:

1. Specify `return_record=True` when calling `InferenceSetup`:
   
```python
result = setup(' ', n_particles=10, return_record=True)
```

2. Save the SMC record in `notes/smc_viz/`:
   
```python
import json
with open('notes/smc_viz/record.json', 'w') as f:
      f.write(json.dumps(result.record))
```

3. Run a server in `notes/smc_viz/`:
   
```bash
python -m http.server --directory notes/smc_viz 8000
```

4. Navigate to [localhost:8000/](http://localhost:8000/).