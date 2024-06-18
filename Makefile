SHELL := /usr/bin/env bash
EXEC = python=3.10
NAME = genparse
RUN = python -m
INSTALL = $(RUN) pip install
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
	@$(INSTALL) -e ".[test]" && pre-commit install

## format    : format code style.
.PHONY : format
format : env
	@ruff format

## docs      : build documentation.
.PHONY : docs
docs : env html/docs/index.html
html/docs/index.html : $(NAME)/*.py
	@pdoc $(NAME) -o $(@D)

## test      : run linting and tests.
.PHONY : test
test: ruff pytest
ruff: env
	@ruff check --fix
pytest : env html/coverage/index.html
html/coverage/index.html : html/pytest/report.html
	@coverage html -d $(@D)
html/pytest/report.html : $(NAME)/*.py tests/*.py
	@coverage run --branch -m pytest --html=$@ --self-contained-html
