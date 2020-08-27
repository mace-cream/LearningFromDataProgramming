#!/bin/bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
export PATH=$PATH:$HOME/.poetry/bin
source $HOME/.poetry/env
poetry install
