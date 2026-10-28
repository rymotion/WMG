#!/usr/bin/env bash
# Put the native pieces into this plugin:
#   1. WmgFeatures.xcframework  (Rust wmg-features, C ABI)  -> darwin/Frameworks/
#   2. WmgDetector.mlmodelc + model_meta.json (Core ML model) -> darwin/wmg_detector/Resources/
# Usage: tool/install.sh <10 s run dir> [<5 s run dir>]
#   e.g. tool/install.sh ../../training/runs/20261024-092802 ../../training/runs/20261024-132328
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
run="${1:?usage: tool/install.sh <10 s run dir> [<5 s run dir>]}"
run="$(cd "$run" && pwd)"
run5="${2:-}"

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
rm -rf "$res/WmgDetector5s.mlmodelc" "$res/model_meta_5s.json"
if [ -n "$run5" ]; then
  run5="$(cd "$run5" && pwd)"
  if [ ! -d "$run5/coreml/WmgDetector.mlmodelc" ]; then
    (cd "$repo/training" && .venv-arm64/bin/python -m wmg_train.export_coreml "$run5/model.pt")
  fi
  cp -R "$run5/coreml/WmgDetector.mlmodelc" "$res/WmgDetector5s.mlmodelc"
  cp "$run5/coreml/model_meta.json" "$res/model_meta_5s.json"
fi
echo "installed features + 10 s model $(basename "$run")${run5:+ + 5 s model $(basename "$run5")} into $here/darwin"
