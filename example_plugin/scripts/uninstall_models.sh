#!/bin/bash
# Uninstallation script for SOTA Talking Head models

set -e

echo "=========================================="
echo "SOTA Talking Head Models Uninstallation"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/external_models"
CONFIG_FILE="$PROJECT_ROOT/sota_benchmarker/config/models_config.yaml"

echo "This will remove all installed SOTA models and their configurations."
echo "Models directory: $MODELS_DIR"
echo ""
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 0
fi

# Function to uninstall a model
uninstall_model() {
    local model_name=$1
    local model_dir=$2
    
    if [ -d "$model_dir" ]; then
        echo "Removing $model_name from $model_dir..."
        rm -rf "$model_dir"
        echo "✓ $model_name removed"
    else
        echo "✗ $model_name not found (skipping)"
    fi
}

# Remove individual models
echo ""
echo "Removing models..."
uninstall_model "Ditto" "$MODELS_DIR/ditto"
uninstall_model "MuseTalk" "$MODELS_DIR/MuseTalk"
uninstall_model "LivePortrait" "$MODELS_DIR/LivePortrait"
uninstall_model "V-Express" "$MODELS_DIR/V-Express"
uninstall_model "Hallo3" "$MODELS_DIR/hallo3"
uninstall_model "Hallo4" "$MODELS_DIR/hallo4"
uninstall_model "EchoMimicV3" "$MODELS_DIR/echomimic_v3"


# Remove external_models directory if empty
if [ -d "$MODELS_DIR" ] && [ -z "$(ls -A $MODELS_DIR)" ]; then
    echo "Removing empty models directory..."
    rmdir "$MODELS_DIR"
fi

# Clean up configuration file
if [ -f "$CONFIG_FILE" ]; then
    echo ""
    read -p "Remove model configurations from $CONFIG_FILE? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$CONFIG_FILE"
        echo "✓ Configuration file removed"
    fi
fi

echo ""
echo "=========================================="
echo "Uninstallation Complete!"
echo "=========================================="
