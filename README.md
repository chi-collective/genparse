![Black](https://github.com/timvieira/genparse/actions/workflows/black.yml/badge.svg)](https://github.com/timvieira/genparse/actions/workflows/black.yml)
![Docs](https://github.com/timvieira/genparse/actions/workflows/docs.yml/badge.svg)](https://github.com/timvieira/genparse/actions/workflows/docs.yml)
![Linter](https://github.com/timvieira/genparse/actions/workflows/pylint.yml/badge.svg)](https://github.com/timvieira/genparse/actions/workflows/pylint.yml)
![Tests](https://github.com/timvieira/genparse/actions/workflows/pytest.yml/badge.svg)](https://github.com/timvieira/genparse/actions/workflows/pytest.yml)

GenParse
========

GenParse is a library equipped with algorithms for intersecting weighted
context-free grammars with large language models.

## Getting Started

This library supports an automated build using [GNU Make](https://www.gnu.org/software/make/).

Requirements: [Conda](https://docs.anaconda.com/free/miniconda/)

```bash
make env # set up env and install dependencies
```

```bash
make docs # build documentation
```

## Contributing

Before pushing a new commit

```bash
make format # run style formatting
```

```bash
make test # run tests
```
