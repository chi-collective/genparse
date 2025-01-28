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

**💡Tip:** GenParse supports grammars with arbitrary regular expressions. In practice, we recommend avoiding extremely permisive regular expressions (e.g., `/.+/`) since these will lead to significantly slower inference. See [issue #62](https://github.com/probcomp/genparse/issues/62).

**💡Tip:** If you don't allow your grammar to generate tokens that begin with a space, generation performance gets much worse. GenParse grammar requires adding a " " terminal to the top-level production rule.

## 2. Create an `InferenceSetup` object

Create an `InferenceSetup` object with your chosen language model and grammar:

```python
setup = InferenceSetup('gpt2', grammar)
```

Find the list of parameters for `InferenceSetup` in the pdocs API documentation [here](https://genparse.gen.dev/api/genparse/util.html#InferenceSetup).


**💡Tip:** To try different grammars without having to instantiate new `InferenceSetup` objects each time, use the `update_grammar` method; `setup.update_grammar(new_grammar)` will replace the existing grammar in `setup` with `new_grammar`.

**💡Tip:** If you choose to use a Llama model you will need to authenticate with huggingface. You can do this by running `huggingface-cli login` and entering your credentials (a token), which you can get from [here](https://huggingface.co/settings/tokens). You will also need to sign a waiver to use the Llama models, which you can do [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B). You may need to wait for approval before you can use the Llama models.


## 3. Run inference

Use the setup object to run SMC inference:

```python
# The result is a ParticleApproximation object
result = setup('Write an SQL query:', n_particles=10, verbosity=1, max_tokens=25)
```

Find the list of parameters for the __call__ method in the pdocs API documentation [here](https://genparse.gen.dev/api/genparse/util.html#InferenceSetup.__call__).

The result from `InferenceSetup` is a `ParticleApproximation` object. This object contains a collection of particles, each representing a generated text sequence. Each particle has two main attributes:

- `context`: The generated text sequence.
- `weight`: A numerical value representing the particle's importance weight. The weights are not normalized probabilities. GenParse provides post-processing to convert these weights into meaningful probabilities, which can be accessed via the `.posterior` property:
```python
>>> result.posterior
{"SELECT name FROM employees GROUP BY name▪" : 1}
```

## 4. Potential functions

**💡 Tip:** Incorporate constraints directly into the grammar when possible, as this will generally improve the quality of inference.

Potential functions can be used to guide generation using additional constraints. A potential function maps (partial) generations to positive real numbers, with higher values indicating a stronger preference for those generations. Intuitively, when applied in SMC, potential functions offer richer signals for resampling steps, allowing computation to be redirected toward more promising particles during the course of generation.

Potentials are provided as input to an `InferenceSetup` call via the `potential` argument and must be defined at the particle beam level. That is, `InferenceSetup` expects potentials to be callables which are provided a *list* of particles as input and return a *list* of log potential values, one for each particle. 

There is an example of a potential function in [genparse_sql_example.py](https://github.com/probcomp/genparse/blob/main/examples/genparse_sql_example.py).

## 5. Visualizing inference

See the [visualizing inference](./visualizing_inference.md) page for more details.
