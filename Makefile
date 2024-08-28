SHELL := /usr/bin/env bash
EXEC = python=3.10
NAME = genparse
TEST = tests
RUN = python -m
INSTALL = $(RUN) pip install
SRC_FILES := $(shell find $(NAME) -name '*.py' -not -path './genparse/experimental/*')
TEST_FILES := $(shell find $(TEST) -name '*.py' -not -path './genparse/experimental/*')
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
.PHONY : env
env : $(NAME).egg-info/
$(NAME).egg-info/ : setup.py
ifeq ("$(wildcard $(_pyproject.toml))","") # ignore pyproject.toml during build, which is only used for rust
	@mv pyproject.toml _pyproject.toml
endif
ifeq ($(shell uname -s),Darwin)
	@$(INSTALL) -e ".[test]" && pre-commit install
else
	@$(INSTALL) -e ".[test,vllm]" && pre-commit install
endif
ifeq ("$(wildcard $(_pyproject.toml))","") # restore so Rust bindings can build later by user separately
	@mv _pyproject.toml pyproject.toml
endif

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
