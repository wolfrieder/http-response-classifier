#!/bin/bash

echo "Creating a virtual environment"
if [ -d env ]; then
    echo "Environment already exists - skip the installment"
else
    python3 -m venv env
fi


echo "Activating the virtual environment"
source env/bin/activate

echo "Installing the requirements"
pip install -r requirements.txt
