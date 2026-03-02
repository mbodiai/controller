#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pip install build
echo "Setup complete! Activate with: source .venv/bin/activate"