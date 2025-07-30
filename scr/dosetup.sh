#!/bin/bash

# This script creates a specific Python venv, installs dependencies, and activates it.

# IMPORTANT: This script must be run with the 'source' command:  source dosetup.sh
VENV_NAME="venv"
PYTHON_VERSION_CMD="python3.11"

# --- Check if the required Python version exists ---
if ! command -v $PYTHON_VERSION_CMD &> /dev/null
then
    echo "Error: The command '$PYTHON_VERSION_CMD' could not be found."
    echo "Please install it with a command appropriate for your system (e.g., 'sudo apt-get install python3.11')."
    return 1
fi

# --- Create the virtual environment ---
echo "Creating virtual environment with $PYTHON_VERSION_CMD..."
$PYTHON_VERSION_CMD -m venv $VENV_NAME

# --- Install packages ---
echo "Installing packages from requirements.txt..."
./${VENV_NAME}/bin/python3 -m pip install -r requirements.txt

# --- Activate the environment ---
echo "Activating virtual environment..."
source ${VENV_NAME}/bin/activate

echo ""
echo "Setup complete. The '${VENV_NAME}' environment is now active."