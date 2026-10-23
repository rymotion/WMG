Pod::Spec.new do |s|
  s.name             = 'wmg_detector'
  s.version          = '0.1.0'
  s.summary          = 'On-device AI-generated-music detector (Rust features + Core ML).'
  s.homepage         = 'https://activeie.org'
  s.license          = { :type => 'Proprietary' }
  s.author           = { 'WMG' => 'ryan@activeie.org' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*.swift'
  s.ios.dependency 'Flutter'
  s.osx.dependency 'FlutterMacOS'
  s.ios.deployment_target = '15.0'
  s.osx.deployment_target = '12.0'
  s.swift_version    = '5.0'
  s.frameworks       = 'CoreML'
  # Static Rust library (wmg-features, built by features/apple/build_xcframework.sh).
  s.vendored_frameworks = 'Frameworks/WmgFeatures.xcframework'
  # Compiled Core ML model + metadata (installed by tool/install.sh from a training run).
  s.resource_bundles = { 'WmgDetectorModel' => ['wmg_detector/Resources/WmgDetector.mlmodelc', 'wmg_detector/Resources/model_meta.json'] }
  # C header + module map for `import WmgFeaturesC` (works for iOS, simulator and macOS slices).
  s.preserve_paths  = ['Frameworks/WmgFeatures.xcframework', 'include/**/*']
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_INCLUDE_PATHS' => '$(PODS_TARGET_SRCROOT)/include',
    'HEADER_SEARCH_PATHS' => '$(PODS_TARGET_SRCROOT)/include',
  }
end
