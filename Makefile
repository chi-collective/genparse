SHELL := /usr/bin/env bash
EXEC = python=3.10
NAME = genparse
RUN = python -m
INSTALL = $(RUN) pip install
ACTIVATE = source activate $(NAME)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## update    : update repo with latest version from GitHub.
.PHONY : update
update :
	@git pull origin

## env       : setup environment and install dependencies.
.PHONY : env
env : $(NAME).egg-info/
$(NAME).egg-info/ : setup.py
ifeq (0, $(shell conda env list | grep -wc $(NAME)))
	@conda create -yn $(NAME) $(EXEC)
endif
	@$(ACTIVATE); $(INSTALL) -e ".[test]"

## format    : format code with black.
.PHONY : format
format : env
	@$(ACTIVATE); black .

## test      : run testing pipeline.
.PHONY : test
test: black pylint pytest
black : env
	@$(ACTIVATE); # black --check .
pylint : env html/pylint/index.html
pytest : env html/coverage/index.html
html/pylint/index.html : html/pylint/index.json
	@$(ACTIVATE) ; pylint-json2html -o $@ -e utf-8 $<
html/pylint/index.json : $(NAME)/*.py
	@mkdir -p $(@D)
	@$(ACTIVATE) ; pylint $(NAME) \
	--disable C0103,C0112,C0113,C0114,C0115,C0116 \
	--output-format=colorized,json:$@ \
	|| pylint-exit $$?
html/coverage/index.html : html/pytest/report.html
	@$(ACTIVATE) ; coverage html -d $(@D)
html/pytest/report.html : $(NAME)/*.py tests/*.py
	@$(ACTIVATE) ; coverage run --branch -m pytest \
	--html=$@ --self-contained-html
