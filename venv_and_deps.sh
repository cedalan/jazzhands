#!/bin/bash
set -e

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv .venv
fi

#Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

#Upgrade pip just in case
echo "Upgrading pip..."
pip install --upgrade pip

#Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Could not find requirements.txt..."
fi

echo "venv ready and requirements installed!"