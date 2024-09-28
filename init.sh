#!/usr/bin/env bash

echo "Check environment ..."
if ! command -v python3 &> /dev/null || ! python3 -c "import venv" &> /dev/null; then
    echo "Either Python 3 is not installed or venv module is missing."
    sudo apt install python3.10.11 python3.10-venv
fi

echo "Creating a virtual environment"
if [ -d env ]; then
    echo "Environment already exists - skip the installment"
else
    python3 -m venv env
fi

echo "Activating the virtual environment"
source env/bin/activate

if ! command -v pip &> /dev/null; then
    echo "Pip not installed."
    sudo apt install python3-pip
fi

python3 -m pip install --upgrade pip

echo "Installing the requirements"
pip install -r requirements.txt
