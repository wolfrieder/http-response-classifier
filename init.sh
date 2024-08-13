#!/bin/bash

echo "Creating a virtual environment"
if [ -d .venv ]; then
    echo "Environment already exists - skip the installment"
else
    python3 -m venv .venv --prompt apa
fi


echo "Activating the virtual environment"
source .venv/bin/activate

echo "Installing the requirements"
pip install -r requirements.txt

deactivate
