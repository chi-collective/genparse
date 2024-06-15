SHELL := /usr/bin/env bash
EXEC = python=3.10
NAME = genparse
RUN = python -m
INSTALL = $(RUN) pip install
ACTIVATE = source activate $(NAME)
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
ifeq (0, $(shell conda env list | grep -wc $(NAME)))
	@conda create -yn $(NAME) $(EXEC)
endif
	@$(ACTIVATE); $(INSTALL) -e ".[test]"

## format    : format code style.
.PHONY : format
format : env
	@$(ACTIVATE); isort . && black .

## docs      : build documentation.
.PHONY : docs
docs : env html/docs/index.html
html/docs/index.html : $(NAME)/*.py
	@$(ACTIVATE); pdoc $(NAME) -o $(@D)

## test      : run linting and tests.
.PHONY : test
test: black pylint pytest
black : env
	@$(ACTIVATE); black --check .
pylint : env html/pylint/index.html
pytest : env html/coverage/index.html
html/pylint/index.html : html/pylint/index.json
	@$(ACTIVATE) ; pylint-json2html -o $@ -e utf-8 $<
html/pylint/index.json : $(NAME)/*.py
	@mkdir -p $(@D)
	@$(ACTIVATE) ; pylint $(NAME) \
	--disable C0103,C0112,C0113,C0114,C0115,C0116,C0301,C0411,C0412,C0413,C0415,C2401,R0902,R0903,R0904,R0912,R0913,R0914 \
	--output-format=colorized,json:$@ \
	|| pylint-exit $$?
html/coverage/index.html : html/pytest/report.html
	@$(ACTIVATE) ; coverage html -d $(@D)
html/pytest/report.html : $(NAME)/*.py tests/*.py
	@$(ACTIVATE) ; coverage run --branch -m pytest \
	--html=$@ --self-contained-html
