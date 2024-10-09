![Docs](https://github.com/timvieira/genparse/actions/workflows/docs.yml/badge.svg)
![Linter](https://github.com/timvieira/genparse/actions/workflows/ruff.yml/badge.svg)
![Tests](https://github.com/timvieira/genparse/actions/workflows/pytest.yml/badge.svg)

# GenParse

GenParse is a sophisticated Python library for constrained text generation.
It combines the power of large language models (like Llama 3.1) with formal grammars to produce text that is both fluent and adheres to specific structural rules. In this library we approximate inference for controlled LLM generation based on Sequential Monte Carlo (SMC). SMC allows us to flexibly incorporate constraints at inference time, and efficiently reallocate computation in light of new information during the course of generation. We utilize VLLM for speed and compute optimization.


## Features

- Integration with popular language models (e.g., Llama 3.1, CodeLlama)
- Custom grammar specification using Lark syntax
- Character-level and token-level proposal mechanisms
- Particle-based sampling for diverse output generation
- Flexible inference setup for various generation tasks

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- make
- git

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/timvieira/genparse.git
   cd genparse
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Set up the environment and install dependencies:
   ```bash
   make env
   ```
   This command will:
   - Install the package in editable mode with test dependencies
   - Install pre-commit hooks

   Note: On macOS, it will install without the 'vllm' extra. On other systems, it will include 'vllm'.

4. Verify the installation:
   ```bash
   python -c "import genparse; print(genparse.__version__)"
   ```

## Quick Start

There is an example script that shows how to use GenParse to generate text from a grammar in genparse_tiny_example.py. You can inspect the program and the output to see a simple example of how to use GenParse. You can run it as follows:

```bash
python genparse_tiny_example.py
```

There is a slighlty more complex text example script in genparse_example.py, and an even more complext example that shows how to use GenParse to generate SQL queries from a grammar in genparse_sql_example.py. You can inspect the program and the output to see a simple example of how to use GenParse to generate SQL queries. You can run it as follows:

```bash
python genparse_sql_example.py
```

## Defining Grammars

GenParse uses the Lark parsing library for grammar specification. For a comprehensive guide on how to write grammars using Lark syntax, please refer to the official Lark documentation:

[Lark Grammar Reference](https://lark-parser.readthedocs.io/en/latest/grammar.html)

## Usage Guide

###0. List Imports
```python
# Import the necessary class from genparse
from genparse import InferenceSetup
```

### 1. Define Your Grammar

Use Lark syntax to define your grammar. For example:

```python
grammar = """
start: statement+
statement: "if" condition "then" action
condition: WORD
action: WORD
WORD: /[a-zA-Z]+/
"""
```

### 2. Set Up Inference

Create an `InferenceSetup` object with your chosen language model and grammar:

```python
setup = InferenceSetup('gpt2', grammar, proposal_name='character')
```

### 3. Generate Text

Use the setup object to generate text:

```python
# The result is a ParticleApproximation object
result = setup("if ", n_particles=10, max_tokens=50)
```

### 4. Process Results

The result from `InferenceSetup` is a `ParticleApproximation` object. This object contains a collection of particles, each representing a possible generated text sequence. Each particle has two main attributes:
- `context`: The generated text sequence.
- `weight`: A numerical value representing the particle's importance or likelihood.

The weights are not normalized probabilities, but rather importance weights. GenParse provides post-processing to convert these weights into meaningful probabilities, which can be accessed via the `.posterior` property. Here's how you can use it:

```python
# Access the posterior approximation
posterior = result.posterior

# Iterate through each unique generated text and its probability
for generated_text, probability in posterior.items():
    # Print the text and its probability
    print(f"'{generated_text}': {probability:.4f}")
```

This code does the following:
1. It accesses the posterior approximation directly from the `ParticleApproximation` object.
2. It iterates through each unique generated text and its calculated probability.
3. Finally, it prints each unique generated text along with its probability.


## Development

After installation, you can use the following commands for development:

- `make help`: Print available commands
- `make update`: Update the repository from GitHub
- `make format`: Format code style using ruff
- `make docs`: Build documentation using pdoc
- `make test`: Run linting (ruff) and tests (pytest with coverage)

## Contributing

Before pushing a new commit, always run:

```bash
make test
```

This will run all tests and ensure code quality.

## Troubleshooting

If you encounter any issues during installation or setup, please try the following:

1. Make sure you ran `make env` to set up your environment.
2. If necessary run `make env` in a fresh environment. 
3. Try running in a virtual environment if you skipped that step.
4. Ensure you have the correct Python version (3.10 or higher)
5. If you encounter any errors, try running `make test` to see more detailed output

If problems persist, please open an issue on our GitHub repository with the error message and your system information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Hugging Face, VLLM, Lark, and all of the teams we have dependencies on.

## Contact

For questions and support, please open an issue on the GitHub repository.
