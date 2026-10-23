#!/usr/bin/env bash
# Put the native pieces into this plugin:
#   1. WmgFeatures.xcframework  (Rust wmg-features, C ABI)  -> darwin/Frameworks/
#   2. WmgDetector.mlmodelc + model_meta.json (Core ML model) -> darwin/wmg_detector/Resources/
# Usage: tool/install.sh <training run dir, e.g. ../../training/runs/20261019-080656>
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
run="${1:?usage: tool/install.sh <training run dir>}"
run="$(cd "$run" && pwd)"

"$repo/features/apple/build_xcframework.sh"
rm -rf "$here/darwin/Frameworks/WmgFeatures.xcframework"
cp -R "$repo/features/apple/WmgFeatures.xcframework" "$here/darwin/Frameworks/"
mkdir -p "$here/darwin/include/WmgFeaturesC"
cp "$repo/features/include/wmg_features.h" "$here/darwin/include/WmgFeaturesC/"

if [ ! -d "$run/coreml/WmgDetector.mlmodelc" ]; then
  (cd "$repo/training" && .venv-arm64/bin/python -m wmg_train.export_coreml "$run/model.pt")
fi
res="$here/darwin/wmg_detector/Resources"
rm -rf "$res/WmgDetector.mlmodelc"
cp -R "$run/coreml/WmgDetector.mlmodelc" "$res/"
cp "$run/coreml/model_meta.json" "$res/"
echo "installed features + model from $(basename "$run") into $here/darwin"
