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
- **model_name** (str): Name of the language model to use. See the main page for the list of models currently supported by GenParse.
- **grammar** (str): The grammar specification in Lark format.

See the docstring for optional arguments that can be provided for more complex usage.

> **💡Tip:** To try different grammars without having to instantiate new `InferenceSetup` objects each time, use the `update_grammar` method; `setup.update_grammar(new_grammar)` will replace the existing grammar in `setup` with `new_grammar`.

### 3. Run inference

Use the setup object to run SMC inference:

```python
# The result is a ParticleApproximation object
result = setup('Write an SQL query:', n_particles=10, verbosity=1, max_tokens=25)
```

When calling `InferenceSetup`, the following arguments are required:
* **prompt** (str): The input prompt to generate samples from.
* **n_particles** (int): The number of particles (samples) to generate.

We also highlight the following optional arguments:
* **max_tokens** (int, optional): The maximum number of tokens to generate. Defaults to 500.
* **verbosity** (int, optional): Verbosity level. When > 0, particles are printed at each step. Default is 0.
* **potential** (Callable, optional): A function that when called on a list of particles, returns a list with the log potential values for each particle. Optional. Potentials can be used to guide generation with additional constraints. See below for an overview of potential functions.
* **ess_threshold** (float, optional): Effective sample size below which resampling is triggered, given as a fraction of **n_particles**. Default is 0.5.


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