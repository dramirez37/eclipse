#!/bin/bash

# This is the main driver script for the project.
# It runs all necessary steps in order: permissions, setup, model, and visuals.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Set Permissions ---
echo "--- Setting permissions for all scripts... ---"
chmod +x doperms.sh
./doperms.sh
echo ""

# --- Set up Environment and Install Packages ---
echo "--- Setting up Python virtual environment and installing packages... ---"
source ./dosetup.sh
echo ""

# --- Run the Model ---
echo "--- Running the main model (final.py)... ---"
python3 final.py
echo ""

# --- Generate All Visuals ---
echo "--- Generating all visual aids (dovisuals.sh)... ---"
./dovisuals.sh
echo ""

echo "--- Workflow Complete! ---"
echo "All files are in the 'output/' directory."