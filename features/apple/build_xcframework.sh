#!/usr/bin/env bash
# Build WmgFeatures.xcframework (static lib + C header) for macOS (arm64 + x86_64), iOS (arm64)
# and the iOS simulator (arm64 + x86_64). Output: features/apple/WmgFeatures.xcframework
set -euo pipefail
cd "$(dirname "$0")/.."
export WMG_FEATURES_COMMIT="${WMG_FEATURES_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
export IPHONEOS_DEPLOYMENT_TARGET=15.0 MACOSX_DEPLOYMENT_TARGET=12.0
targets=(aarch64-apple-darwin x86_64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios)
for t in "${targets[@]}"; do
  "$CARGO" build --release --lib --features ffi --target "$t"
done
out=apple/WmgFeatures.xcframework
rm -rf "$out" apple/headers apple/universal
mkdir -p apple/headers apple/universal/macos apple/universal/ios-sim
# Header only: consumers (e.g. the wmg_detector plugin) provide their own module map, so the
# module isn't defined twice.
cp include/wmg_features.h apple/headers/
# One slice per platform: macOS and the simulator are universal (arm64 + x86_64), since Xcode
# picks a slice by platform and requires every architecture being built.
lib=libwmg_features.a
lipo -create target/aarch64-apple-darwin/release/$lib target/x86_64-apple-darwin/release/$lib \
  -output apple/universal/macos/$lib
lipo -create target/aarch64-apple-ios-sim/release/$lib target/x86_64-apple-ios/release/$lib \
  -output apple/universal/ios-sim/$lib
args=(-library apple/universal/macos/$lib -headers apple/headers
      -library target/aarch64-apple-ios/release/$lib -headers apple/headers
      -library apple/universal/ios-sim/$lib -headers apple/headers)
xcodebuild -create-xcframework "${args[@]}" -output "$out"
echo "built $out (commit $WMG_FEATURES_COMMIT)"
