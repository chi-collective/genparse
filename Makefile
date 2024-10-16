SHELL := /usr/bin/env bash
EXEC = python=3.10
NAME = genparse
TEST = tests
RUN = python -m
INSTALL = $(RUN) pip install
SRC_FILES := $(shell find $(NAME) -name '*.py')
TEST_FILES := $(shell find $(TEST) -name '*.py')
.DEFAULT_GOAL := help

## help      : print available commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repository from GitHub.
.PHONY : update
update :
	@git pull origin

## env       : setup env and install dependencies.
## Usage:
## make env       : Set up the environment and build the Rust parser
## make env-no-rust : Set up the environment without building the Rust parser
.PHONY : env env-no-rust
env : $(NAME).egg-info/
env-no-rust : $(NAME).egg-info/
$(NAME).egg-info/ : setup.py
# check if rust is installed
	@if [ "$@" = "env" ]; then \
		if ! command -v rustc > /dev/null; then \
			echo "GenParse optionally depends on Rust but it is not installed. Please install Rust from https://www.rust-lang.org/tools/install"; \
			echo "You can check if Rust is installed by running 'rustc --version'"; \
			echo "If you don't want to install Rust, you can use 'make env-no-rust' instead"; \
			exit 1; \
		fi \
	fi
# temporarily move pyproject.toml to avoid conflicts with setup.py
	@if [ -f pyproject.toml ]; then \
		mv pyproject.toml pyproject.toml.bak; \
	fi
# install dependencies from setup.py with pre-commit, while ignoring pyproject.toml
	@( \
		trap 'status=$$?; if [ -f pyproject.toml.bak ]; then mv pyproject.toml.bak pyproject.toml; fi; exit $$status' EXIT; \
		set -e; \
		if [ "$$(uname -s)" = "Darwin" ]; then \
			$(INSTALL) -e ".[test]" && pre-commit install; \
		else \
			$(INSTALL) -e ".[test,vllm]" && pre-commit install; \
		fi \
	)
# build rust parser (only for 'env' target)
	@if [ "$@" = "env" ]; then \
		maturin develop --release; \
	fi

## format    : format code style.
.PHONY : format
format : env
	@ruff format

## docs      : build documentation.
.PHONY : docs
docs : env html/docs/index.html
html/docs/index.html : $(SRC_FILES)
	@pdoc $(NAME) --math -o $(@D)

## test      : run linting and tests.
.PHONY : test
test : ruff pytest
ruff : env
	@ruff check --fix
pytest : env html/coverage/index.html
html/coverage/index.html : html/pytest/report.html
	@coverage html -d $(@D)
html/pytest/report.html : $(SRC_FILES) $(TEST_FILES)
	@coverage run --branch -m pytest --html=$@ --self-contained-html $(SRC_FILES) $(TEST_FILES)
