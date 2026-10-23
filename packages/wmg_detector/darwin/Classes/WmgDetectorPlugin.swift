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
///   info  -> {featureSpecHash, modelSpecHash, modelRun, headWeights: Float64List, headBias}
///   score {pcm: Int16 LE bytes, sampleRate}
///         -> {inputRmsDbfs, windows: [{startS, logit, bands, embedding: Float32List}]}
/// Features come from the Rust wmg-features library (identical to training); the model is the
/// compiled Core ML BandNet bundled in this plugin. Work runs off the main thread.
public class WmgDetectorPlugin: NSObject, FlutterPlugin {
  private let queue = DispatchQueue(label: "org.activeie.wmg_detector", qos: .userInitiated)
  private var model: MLModel?
  private var meta: [String: Any] = [:]

  private static let maxBatch = 16

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
        switch call.method {
        case "info":
          try self.ensureModel()
          reply = [
            "featureSpecHash": String(cString: wmg_spec_hash()),
            "modelSpecHash": self.meta["feature_spec_hash"] as? String ?? "",
            "modelRun": self.meta["model_run"] as? String ?? "",
            "headWeights": FlutterStandardTypedData(float64: Self.float64Data(self.meta["head_weights"] as? [Double] ?? [])),
            "headBias": self.meta["head_bias"] as? Double ?? 0,
          ]
        case "score":
          guard let args = call.arguments as? [String: Any],
                let pcm = args["pcm"] as? FlutterStandardTypedData,
                let sampleRate = args["sampleRate"] as? Int else {
            throw DetectorError.badArguments
          }
          reply = try self.score(pcm16: pcm.data, sampleRate: sampleRate)
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

  // MARK: - Model

  private func resourceBundle() throws -> Bundle {
    let pluginBundle = Bundle(for: WmgDetectorPlugin.self)
    if let url = pluginBundle.url(forResource: "WmgDetectorModel", withExtension: "bundle"),
       let b = Bundle(url: url) {
      return b
    }
    return pluginBundle
  }

  private func ensureModel() throws {
    if model != nil { return }
    let bundle = try resourceBundle()
    guard let url = bundle.url(forResource: "WmgDetector", withExtension: "mlmodelc") else {
      throw DetectorError.modelMissing
    }
    if let metaURL = bundle.url(forResource: "model_meta", withExtension: "json"),
       let data = try? Data(contentsOf: metaURL),
       let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
      meta = obj
    }
    let expected = meta["feature_spec_hash"] as? String ?? ""
    let actual = String(cString: wmg_spec_hash())
    guard expected == actual else { throw DetectorError.specMismatch(model: expected, features: actual) }
    let config = MLModelConfiguration()
    config.computeUnits = .all
    model = try MLModel(contentsOf: url, configuration: config)
  }

  // MARK: - Scoring

  private func score(pcm16: Data, sampleRate: Int) throws -> [String: Any] {
    try ensureModel()
    guard let model else { throw DetectorError.modelMissing }

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

    let nMels = Int(wmg_n_mels()), frames = Int(wmg_frames_per_window())
    let per = nMels * frames
    let maxWindows = wmg_max_windows(n, UInt32(sampleRate))
    guard maxWindows > 0 else { return ["inputRmsDbfs": inputRmsDbfs, "windows": [Any]()] }
    var mel = [Float](repeating: 0, count: maxWindows * per)
    var starts = [Float](repeating: 0, count: maxWindows)
    let got = samples.withUnsafeBufferPointer { s in
      mel.withUnsafeMutableBufferPointer { m in
        starts.withUnsafeMutableBufferPointer { st in
          wmg_extract_windows(s.baseAddress, n, UInt32(sampleRate), 1, m.baseAddress, st.baseAddress, maxWindows)
        }
      }
    }
    guard got >= 0 else { throw DetectorError.features }

    var windows: [[String: Any]] = []
    var i = 0
    while i < Int(got) {
      let b = min(Self.maxBatch, Int(got) - i)
      let input = try MLMultiArray(shape: [NSNumber(value: b), 1, NSNumber(value: nMels), NSNumber(value: frames)],
                                   dataType: .float32)
      mel.withUnsafeBufferPointer { src in
        let dst = input.dataPointer.bindMemory(to: Float.self, capacity: b * per)
        dst.update(from: src.baseAddress! + i * per, count: b * per)
      }
      let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["logmel": input]))
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
          "startS": Double(starts[i + j]),
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
  case badArguments, modelMissing, features, output
  case specMismatch(model: String, features: String)

  var description: String {
    switch self {
    case .badArguments: return "score needs {pcm: Uint8List, sampleRate: int}"
    case .modelMissing: return "WmgDetector.mlmodelc not bundled; run packages/wmg_detector/tool/install.sh"
    case .features: return "feature extraction failed"
    case .output: return "model output missing ai_logit/band_scores/embedding (re-export with export_coreml)"
    case let .specMismatch(model, features):
      return "model was trained on feature spec \(model) but this app's wmg-features is \(features)"
    }
  }
}
