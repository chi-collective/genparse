![Docs](https://github.com/timvieira/genparse/actions/workflows/docs.yml/badge.svg)
![Linter](https://github.com/timvieira/genparse/actions/workflows/ruff.yml/badge.svg)
![Tests](https://github.com/timvieira/genparse/actions/workflows/pytest.yml/badge.svg)

GenParse
========

GenParse is a library equipped with algorithms for intersecting weighted
context-free grammars with large language models.

## Getting Started

This library supports an automated build using [GNU Make](https://www.gnu.org/software/make/).

```bash
make env # install dependencies in current env
```

```python
>>> from genparse import InferenceSetup
>>> grammar = """
... start: "Sequential Monte Carlo is " ( "good" | "bad" )
... """
>>> m = InferenceSetup('gpt2', grammar, proposal_name='character')
>>> m(' ', n_particles=15)
{
  'Sequential Monte Carlo is good▪': 0.7770842914205952,
  'Sequential Monte Carlo is bad▪': 0.22291570857940482,
}
```

## Contributing

Before pushing a new commit

```bash
make test # run tests
```
