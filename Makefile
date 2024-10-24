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
##   Usage:
##   make env            : Set up the environment and build the Rust parser
##   make env-no-rust    : Set up the environment without building the Rust parser
.PHONY : env env-no-rust
env : $(NAME).egg-info/
env-no-rust : $(NAME).egg-info/
$(NAME).egg-info/ : setup.py
# check if rust is installed
# rustc --version
	@if [ "$(MAKECMDGOALS)" = "env" ]; then \
		if ! command -v rustc > /dev/null; then \
			echo "GenParse optionally depends on Rust for faster Earley parsing, but it is not installed."; \
			echo "Please install Rust from https://www.rust-lang.org/tools/install or use 'make env-no-rust' to install GenParse without it"; \
			exit 1; \
		fi \
	else \
		echo "Skipping Rust parser installation. Fast Earley parsing with Rust will not be available."; \
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
			echo "Skipping vllm installation on macOS. GPU-accelerated inference with vllm will not be available."; \
			$(INSTALL) -e ".[test]" && pre-commit install; \
		else \
			$(INSTALL) -e ".[test,vllm]" && pre-commit install; \
		fi \
	)
# build rust parser (only for 'env' target)
	@if [ "$(MAKECMDGOALS)" = "env" ]; then \
		echo "Building rust parser"; \
		maturin develop --release; \
	fi

##   refresh_env         : Force refresh the environment setup with Rust.
##   refresh_env-no-rust : Force refresh the environment setup without Rust.
.PHONY : refresh_env refresh_env-no-rust
refresh_env :
	@rm -rf $(NAME).egg-info/
	@$(MAKE) env

refresh_env-no-rust :
	@rm -rf $(NAME).egg-info/
	@$(MAKE) env-no-rust
	
## format    : format code style.
.PHONY : format
format : env
	@ruff format

## docs      : build documentation.
.PHONY : docs
docs : env html/docs/index.html
html/docs/index.html : $(SRC_FILES)
	@pdoc $(NAME) --docformat google --math -o $(@D)

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
