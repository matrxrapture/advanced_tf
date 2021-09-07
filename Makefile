
SHELL := /bin/bash
VENV_ACTIVATE := .venv/bin/activate
VENV_DEACTIVATE := .venv/bin/deactivate
PYTHON := python3
${VENV_ACTIVATE}: requirements.txt
	python3.9 -m venv .venv || python3 -m venv .venv
	source ${VENV_ACTIVATE} && python3 -m pip install --upgrade pip setuptools && python3 -m pip install -r requirements.txt


# .PHONY: all init 

# docker/generated.mk: docker/generate_makefile.py docker/image_types.yaml fuzzers benchmarks ${VENV_ACTIVATE}
	source ${VENV_ACTIVATE} && PYTHONPATH=. python3 $< $@

init:
	@echo "start log util"
	export PATH=/home/elton/.local/bin:$PATH
install-dependencies:
	@echo "Installing requirements and activating env"
	 ${VENV_ACTIVATE}


lint: install-dependencies
	@echo "Installing requirements"

	source ${VENV_ACTIVATE} && ${PYTHON} main.py lint

run-main-file: 
	source ${VENV_ACTIVATE} && ${PYTHON} main.py

clean:
	@echo "removing the env"
	rm -rvf ./.venv

all: init install-dependencies lint run-mainfile