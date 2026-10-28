import CoreML
import Foundation
// Implementation-only: apps importing this plugin don't need to see the C module.
@_implementationOnly import WmgFeaturesC

#if os(iOS)
import Flutter
#elseif os(macOS)
import FlutterMacOS
#endif

/// Method channel `wmg_detector`:
///   info  {model?: "10s" | "5s"}
///         -> {featureSpecHash, modelSpecHash, modelRun, windowSeconds, headWeights: Float64List, headBias}
///   score {pcm: Int16 LE bytes, sampleRate, model?: "10s" | "5s"}
///         -> {inputRmsDbfs, windows: [{startS, logit, bands, embedding: Float32List}]}
/// Features come from the Rust wmg-features library (identical to training). Two compiled Core ML
/// BandNets can be bundled: `WmgDetector` (10 s windows) and `WmgDetector5s` (5 s windows).
/// The 5 s model scores halves of the library's 10 s windows, exactly as it was trained.
/// Work runs off the main thread.
public class WmgDetectorPlugin: NSObject, FlutterPlugin {
  private let queue = DispatchQueue(label: "org.activeie.wmg_detector", qos: .userInitiated)
  private var models: [String: LoadedModel] = [:]

  private static let maxBatch = 16
  private static let variants: [String: (resource: String, meta: String)] = [
    "10s": ("WmgDetector", "model_meta"),
    "5s": ("WmgDetector5s", "model_meta_5s"),
  ]

  struct LoadedModel {
    let model: MLModel
    let meta: [String: Any]
    /// Frames per model input window: 1000 (10 s) or 500 (5 s).
    let frames: Int
  }

  public static func register(with registrar: FlutterPluginRegistrar) {
    #if os(iOS)
    let messenger = registrar.messenger()
    #else
    let messenger = registrar.messenger
    #endif
    let channel = FlutterMethodChannel(name: "wmg_detector", binaryMessenger: messenger)
    registrar.addMethodCallDelegate(WmgDetectorPlugin(), channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    queue.async {
      let reply: Any
      do {
        let args = call.arguments as? [String: Any] ?? [:]
        let variant = args["model"] as? String ?? "10s"
        switch call.method {
        case "info":
          let m = try self.load(variant)
          reply = [
            "featureSpecHash": String(cString: wmg_spec_hash()),
            "modelSpecHash": m.meta["feature_spec_hash"] as? String ?? "",
            "modelRun": m.meta["model_run"] as? String ?? "",
            "windowSeconds": Double(m.frames) / 100,
            "headWeights": FlutterStandardTypedData(float64: Self.float64Data(m.meta["head_weights"] as? [Double] ?? [])),
            "headBias": m.meta["head_bias"] as? Double ?? 0,
          ]
        case "score":
          guard let pcm = args["pcm"] as? FlutterStandardTypedData,
                let sampleRate = args["sampleRate"] as? Int else {
            throw DetectorError.badArguments
          }
          reply = try self.score(pcm16: pcm.data, sampleRate: sampleRate, model: try self.load(variant))
        default:
          DispatchQueue.main.async { result(FlutterMethodNotImplemented) }
          return
        }
      } catch {
        let err = FlutterError(code: "detector_error", message: "\(error)", details: nil)
        DispatchQueue.main.async { result(err) }
        return
      }
      DispatchQueue.main.async { result(reply) }
    }
  }

  private static func float64Data(_ v: [Double]) -> Data {
    v.withUnsafeBufferPointer { Data(buffer: $0) }
  }

  // MARK: - Models

  private func resourceBundle() -> Bundle {
    let pluginBundle = Bundle(for: WmgDetectorPlugin.self)
    if let url = pluginBundle.url(forResource: "WmgDetectorModel", withExtension: "bundle"),
       let b = Bundle(url: url) {
      return b
    }
    return pluginBundle
  }

  private func load(_ variant: String) throws -> LoadedModel {
    if let m = models[variant] { return m }
    guard let names = Self.variants[variant] else { throw DetectorError.unknownModel(variant) }
    let bundle = resourceBundle()
    guard let url = bundle.url(forResource: names.resource, withExtension: "mlmodelc") else {
      throw DetectorError.modelMissing(names.resource)
    }
    var meta: [String: Any] = [:]
    if let metaURL = bundle.url(forResource: names.meta, withExtension: "json"),
       let data = try? Data(contentsOf: metaURL),
       let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
      meta = obj
    }
    let expected = meta["feature_spec_hash"] as? String ?? ""
    let actual = String(cString: wmg_spec_hash())
    guard expected == actual else { throw DetectorError.specMismatch(model: expected, features: actual) }
    let config = MLModelConfiguration()
    config.computeUnits = .all
    let loaded = LoadedModel(model: try MLModel(contentsOf: url, configuration: config),
                             meta: meta,
                             frames: (meta["frames"] as? Int) ?? Int(wmg_frames_per_window()))
    models[variant] = loaded
    return loaded
  }

  // MARK: - Scoring

  private func score(pcm16: Data, sampleRate: Int, model m: LoadedModel) throws -> [String: Any] {
    let n = pcm16.count / 2
    var samples = [Float](repeating: 0, count: n)
    pcm16.withUnsafeBytes { raw in
      let p = raw.bindMemory(to: Int16.self)
      for i in 0..<n { samples[i] = Float(Int16(littleEndian: p[i])) / 32768 }
    }
    var sumSq: Double = 0
    for v in samples { sumSq += Double(v) * Double(v) }
    let rms = n > 0 ? sqrt(sumSq / Double(n)) : 0
    let inputRmsDbfs = rms > 0 ? 20 * log10(rms) : -200

    // Library windows: 10 s, 50% overlap, level-normalised, silence-gated.
    let nMels = Int(wmg_n_mels()), fullFrames = Int(wmg_frames_per_window())
    let per = nMels * fullFrames
    let maxWindows = wmg_max_windows(n, UInt32(sampleRate))
    guard maxWindows > 0 else { return ["inputRmsDbfs": inputRmsDbfs, "windows": [Any]()] }
    var mel = [Float](repeating: 0, count: maxWindows * per)
    var starts = [Float](repeating: 0, count: maxWindows)
    let got = samples.withUnsafeBufferPointer { s in
      mel.withUnsafeMutableBufferPointer { mm in
        starts.withUnsafeMutableBufferPointer { st in
          wmg_extract_windows(s.baseAddress, n, UInt32(sampleRate), 1, mm.baseAddress, st.baseAddress, maxWindows)
        }
      }
    }
    guard got >= 0 else { throw DetectorError.features }

    // Model inputs: whole windows, or for a shorter model, slices of them (halves for 5 s).
    // Slices that start beyond the audio (zero padding of a short clip) and repeats from the
    // 50% overlap are skipped, as in training.
    let frames = m.frames
    let parts = fullFrames / frames
    let durationS = Double(n) / Double(sampleRate)
    let sliceS = Double(frames) / 100
    var inputs: [(start: Double, src: Int, part: Int)] = []
    var seen = Set<Int>()
    for w in 0..<Int(got) {
      for p in 0..<parts {
        let start = Double(starts[w]) + Double(p) * sliceS
        let key = Int((start * 100).rounded())
        if seen.contains(key) { continue }
        if p > 0 && start + sliceS > durationS + 0.05 { continue }
        seen.insert(key)
        inputs.append((start, w, p))
      }
    }

    var windows: [[String: Any]] = []
    var i = 0
    while i < inputs.count {
      let b = min(Self.maxBatch, inputs.count - i)
      let input = try MLMultiArray(shape: [NSNumber(value: b), 1, NSNumber(value: nMels), NSNumber(value: frames)],
                                   dataType: .float32)
      mel.withUnsafeBufferPointer { src in
        let dst = input.dataPointer.bindMemory(to: Float.self, capacity: b * nMels * frames)
        for j in 0..<b {
          let (_, w, p) = inputs[i + j]
          for r in 0..<nMels {  // row-major [mel][frame]
            (dst + (j * nMels + r) * frames).update(from: src.baseAddress! + w * per + r * fullFrames + p * frames,
                                                     count: frames)
          }
        }
      }
      let out = try m.model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["logmel": input]))
      guard let logits = out.featureValue(for: "ai_logit")?.multiArrayValue,
            let bands = out.featureValue(for: "band_scores")?.multiArrayValue,
            let emb = out.featureValue(for: "embedding")?.multiArrayValue else {
        throw DetectorError.output
      }
      let dim = emb.shape.last!.intValue
      for j in 0..<b {
        var e = [Float](repeating: 0, count: dim)
        for k in 0..<dim { e[k] = emb[[NSNumber(value: j), NSNumber(value: k)]].floatValue }
        windows.append([
          "startS": inputs[i + j].start,
          "logit": logits[j].doubleValue,
          "bands": (0..<3).map { bands[[NSNumber(value: j), NSNumber(value: $0)]].doubleValue },
          "embedding": FlutterStandardTypedData(float32: e.withUnsafeBufferPointer { Data(buffer: $0) }),
        ])
      }
      i += b
    }
    return ["inputRmsDbfs": inputRmsDbfs, "windows": windows]
  }
}

enum DetectorError: Error, CustomStringConvertible {
  case badArguments, features, output
  case unknownModel(String)
  case modelMissing(String)
  case specMismatch(model: String, features: String)

  var description: String {
    switch self {
    case .badArguments: return "score needs {pcm: Uint8List, sampleRate: int, model?: '10s'|'5s'}"
    case let .unknownModel(v): return "unknown model '\(v)' (expected 10s or 5s)"
    case let .modelMissing(name): return "\(name).mlmodelc not bundled; run packages/wmg_detector/tool/install.sh"
    case .features: return "feature extraction failed"
    case .output: return "model output missing ai_logit/band_scores/embedding (re-export with export_coreml)"
    case let .specMismatch(model, features):
      return "model was trained on feature spec \(model) but this app's wmg-features is \(features)"
    }
  }
}
