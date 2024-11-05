![Docs](https://github.com/timvieira/genparse/actions/workflows/docs.yml/badge.svg)
![Linter](https://github.com/timvieira/genparse/actions/workflows/ruff.yml/badge.svg)
![Tests](https://github.com/timvieira/genparse/actions/workflows/pytest.yml/badge.svg)

# GenParse

GenParse is a Python library for constrained generation with language models, specialized for tasks like semantic parsing and code generation. It uses sequential Monte Carlo (SMC) inference to ensure that language model generations comply with user-defined syntactic and semantic constraints. The library is equipped with proposal distributions that efficiently enforce syntactic constraints, supports constraints beyond grammaticality through arbitrary scoring (*potential*) functions, and is integrated with [vLLM](https://docs.vllm.ai/en/latest/) for fast language model inference.


> **⚠️ Warning:** This library is currently in active development. We recommend frequently pulling the latest version to stay updated with improvements and bug fixes. Please report any bugs in [the issue tracker](https://github.com/probcomp/genparse/issues).

## Installation

This library supports an automated build using [GNU Make](https://www.gnu.org/software/make/).

### Prerequisites

- Python 3.10 - 3.12
- pip (Python package installer)
- make
- git
- A GPU with compute capability 7.0 or higher (GPUs are not required, but strongly recommended)

### Steps

1. Clone this repository:
   ```bash
   git clone git@github.com:probcomp/genparse.git
   cd genparse
   ```
2. Create and activate a virtual environment. Using Conda (recommended):
   ```bash
   conda create -n genparse python=3.10
   conda activate genparse
   ```
   Using Python's `venv` module:
   ```bash
   python -m venv genparse
   source genparse/bin/activate  # On Windows, use `genparse\Scripts\activate`
   ```

3. Install package in editable mode with pre-commit hooks
   ```bash
   make env 
   ```
   GenParse optionally depends on Rust for faster parsing. If you do not have Rust installed, you will prompted to do so. However, if you do not want to install Rust, you can also install the library without the Rust dependency via:
   ```bash
   make env-no-rust
   ```

4. You can test your installation by running the following example:
   ```python
   >>> from genparse import InferenceSetup
   >>> grammar = 'start: "Sequential Monte Carlo is " ( "good" | "bad" )'
   >>> m = InferenceSetup('gpt2', grammar, proposal_name='character')
   >>> m(' ', n_particles=15)
   {
     'Sequential Monte Carlo is good▪': 0.7770842914205952,
     'Sequential Monte Carlo is bad▪': 0.22291570857940482,
   }
   ```
   Or simply by running `python genparse_tiny_example.py`.

## Usage Guide

GenParse currently provides a high-level interface for constrained generation via the `InferenceSetup` class. We recommend using this class as its internals may be deprecated without prior warning. 

```python
from genparse import InferenceSetup
```

### 1. Define your grammar 

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

### 2. Create an `InferenceSetup` object

Create an `InferenceSetup` object with your chosen language model and grammar:

```python
setup = InferenceSetup('gpt2', grammar)
```

`InferenceSetup` requires the following arguments:
- **model_name** (str): Name of the language model to use. See [Supported language models](#Supported-language-models) for the list of models currently supported by GenParse.
- **grammar** (str): The grammar specification in Lark format.

`InferenceSetup` accepts the following optional arguments:
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



### 3. Run inference

Use the setup object to run SMC inference:

```python
# The result is a ParticleApproximation object
result = setup('Write an SQL query:', n_particles=10, verbosity=1, max_tokens=25)
```

When calling `InferenceSetup`, the following arguments are required:
- **prompt** (str): The input prompt to generate samples from. This is the starting text for the language model.
- **n_particles** (int): The number of particles (samples) to generate.

We also highlight the following optional arguments:
- **method** (str): The sampling method to use. Options include 'smc' for Sequential Monte Carlo and 'is' for importance sampling. Default to 'smc'.
* **max_tokens** (int): The maximum number of tokens to generate. Defaults to 500.

The following optional arguments are passed in as **kwargs**, which may be expanded over time.
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

### 4. Potential functions

> **💡 Tip:** Incorporate constraints directly into the grammar when possible, as this will generally improve the quality of inference.

Potential functions can be used to guide generation using additional constraints. A potential function maps (partial) generations to positive real numbers, with higher values indicating a stronger preference for those generations. Intuitively, when applied in SMC, potential functions offer richer signals for resampling steps, allowing computation to be redirected toward more promising particles during the course of generation.

Potentials are provided as input to an `InferenceSetup` call via the `potential` argument and must be defined at the particle beam level. That is, `InferenceSetup` expects potentials to be callables which are provided a *list* of particles as input and return a *list* of log potential values, one for each particle. 

### 5. Visualizing inference

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

## Supported language models

Genparse currently supports the following HuggingFace language models. If you would like support for an additional model, please create an issue. 

| Name              | HuggingFace Identifier               |
|-------------------|--------------------------------------|
| llama3            | meta-llama/Meta-Llama-3-8B           |
| llama3.1          | meta-llama/Meta-Llama-3.1-8B         |
| llama3-instruct   | meta-llama/Meta-Llama-3-8B-Instruct  |
| llama3.1-instruct | meta-llama/Meta-Llama-3.1-8B-Instruct|
| codellama         | codellama/CodeLlama-7b-Instruct-hf   |
| gpt2              | gpt2                                 |
| gpt2-medium       | gpt2-medium                          |
| gpt2-large        | gpt2-large                           |

> **💡Tip**: Adding a `mock-` prefix to a language model name will create an imitation language model over the same vocabulary that can be used for testing (e.g., `mock-gpt2`). In practice, these models can be useful for rapid prototyping with minimal hardware.


## Development

After installation, you can use the following commands for development:

- `make help`: Print available commands
- `make update`: Update the repository from GitHub
- `make format`: Format code style using ruff
- `make docs`: Build documentation using pdoc
- `make test`: Run linting (ruff) and tests (pytest with coverage)

After you run make docs, the documentation will be built and output to the html/docs/index.html file. You can open this file in your browser to view the generated documentation.

## Contributing

Before pushing a new commit, always run:

```bash
make test
```

This will run all tests and ensure code quality.

## Troubleshooting

If you encounter any issues during installation or setup, please try the following:

1. Check the common issues below.
2. Make sure you ran `make env` to set up your environment.
3. If necessary run `make env` in a fresh environment. 
4. Try running in a virtual environment if you skipped that step.
5. Ensure you have the correct Python version (3.10 - 3.12).
6. If you encounter any errors, try running `make test` to see more detailed output.

If problems persist, please open an issue on our GitHub repository with the error message and your system information.

### Common issues

- Running `make env` outputs `make: Nothing to be done for 'env'.`
   - Run `make refresh_env` (or `make refresh_env-no-rust`) to force refresh the environment.
- If you are getting `RuntimeError: CUDA error: no kernel image is available for execution on the device` or `UserWarning: CUDA initialization: CUDA unknown error`, you may be using a GPU that is incompatable with `vLLM`. See [the vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html) for GPU requirements.
- If you are getting `TypeError: log_sample() got an unexpected keyword argument 'size'`, you have the wrong version of `arsenal` installed. Create a fresh environment and reinstall `genparse`.
- If you are getting `UserWarning: Failed to initialize NumPy: _ARRAY_API not found` with text
  
   ```
   A module that was compiled using NumPy 1.x cannot be run in
   NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
   versions of NumPy, modules must be compiled with NumPy 2.0.
   Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
   
   If you are a user of the module, the easiest solution will be to
   downgrade to 'numpy<2' or try to upgrade the affected module.
   We expect that some modules will need time to support NumPy 2.
   ```
   then you should downgrade your version of numpy via `pip install "numpy<2"`.


## Reference Papers
An Introduction to Sequential Monte Carlo Language Model Probabilistic Programming framework of Lew et al. (2023)

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to VLLM, Lark, Hugging Face and all of the teams we have dependencies on.

## Contact

For questions and support, please open an issue on the GitHub repository.
