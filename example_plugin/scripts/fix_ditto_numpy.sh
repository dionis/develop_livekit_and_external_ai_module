#!/bin/bash
# Fix NumPy 2.0 compatibility in Ditto by replacing deprecated np.atan2 with np.arctan2

set -e

echo "=========================================="
echo "Fixing NumPy 2.0 Compatibility in Ditto"
echo "=========================================="

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DITTO_DIR="$PROJECT_ROOT/external_models/ditto-talkinghead"

# Check if Ditto exists
if [ ! -d "$DITTO_DIR" ]; then
    echo "❌ Ditto not found at $DITTO_DIR"
    echo "   Please install Ditto first using:"
    echo "   bash sota_benchmarker/scripts/install_models/install_ditto.sh"
    exit 1
fi

echo "Ditto directory: $DITTO_DIR"
echo ""

# Search for files with np.atan2
echo "Searching for files with np.atan2..."
FILES_WITH_ATAN2=$(grep -rl "np\.atan2" "$DITTO_DIR" --include="*.py" || true)

if [ -z "$FILES_WITH_ATAN2" ]; then
    echo "✓ No files need fixing (already compatible)"
    exit 0
fi

echo "Found files to fix:"
echo "$FILES_WITH_ATAN2" | while read file; do
    echo "  - $(realpath --relative-to="$DITTO_DIR" "$file")"
done
echo ""

# Apply fix
echo "Applying fixes..."
echo "$FILES_WITH_ATAN2" | while read file; do
    if [ -f "$file" ]; then
        sed -i 's/np\.atan2/np.arctan2/g' "$file"
        echo "  ✓ Fixed: $(realpath --relative-to="$DITTO_DIR" "$file")"
    fi
done

echo ""
echo "=========================================="
echo "✅ SUCCESS: All files patched"
echo "=========================================="

# Verification
echo ""
echo "Verifying fixes..."
REMAINING=$(grep -r "np\.atan2" "$DITTO_DIR" --include="*.py" || true)

if [ -z "$REMAINING" ]; then
    echo "✓ No remaining np.atan2 found - patch successful!"
else
    echo "⚠️  Warning: Some occurrences may remain:"
    echo "$REMAINING"
fi

echo ""
echo "Next steps:"
echo "  1. Run benchmark:"
echo "     python sota_benchmarker/benchmark_runner.py --model ditto --precision FP16"
echo "  2. If you see other NumPy errors, run this script again"
