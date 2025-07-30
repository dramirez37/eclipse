#!/bin/bash

# --- Generate plots from Python ---
echo "Generating Python visuals (AUC, Importance, Confusion Matrix, Lift)..."
python3 generate_visuals.py
echo "Python visuals saved to the 'output/' directory."
echo ""

# --- Generate the model architecture diagram ---
echo "Generating model architecture diagram from .dot file..."
dot -Tsvg -o output/model_architecture.svg model_architecture.dot
echo "Architecture diagram saved to 'output/model_architecture.svg'."
echo ""

echo "All visuals have been successfully generated."
