![Docs](https://github.com/timvieira/genparse/actions/workflows/docs.yml/badge.svg)
![Linter](https://github.com/timvieira/genparse/actions/workflows/ruff.yml/badge.svg)
![Tests](https://github.com/timvieira/genparse/actions/workflows/pytest.yml/badge.svg)

# GenParse

GenParse is a Python library for constrained generation with language models, specialized for tasks like semantic parsing and code generation. It uses sequential Monte Carlo (SMC) inference to ensure that language model generations comply with user-defined syntactic and semantic constraints. The library is equipped with proposal distributions that efficiently enforce syntactic constraints, supports constraints beyond grammaticality through arbitrary scoring (*potential*) functions, and is integrated with [vLLM](https://docs.vllm.ai/en/latest/) for fast language model inference.


> **⚠️ Warning:** This library is currently in active development. We recommend frequently pulling the latest version to stay updated with improvements and bug fixes. Please report any bugs in [the issue tracker](https://github.com/probcomp/genparse/issues).

First time here? Go to our [Full Documentation](docs/index.md).

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
- `make mkdocs`: Build full documentation using mkdocs
- `make test`: Run linting (ruff) and tests (pytest with coverage)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to VLLM, Lark, Hugging Face and all of the teams we have dependencies on.

## Contact

For questions and support, please open an issue on the GitHub repository.
