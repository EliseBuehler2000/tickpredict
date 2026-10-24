// Checks the app's Swift inference against the Python release pipeline.
//
// Usage: parity-check FIXTURES_DIR COMPILED_MODELS_DIR
// FIXTURES_DIR comes from scripts/ios_parity_fixtures.py; COMPILED_MODELS_DIR holds
// TickPredictV2.mlmodelc and EngorgementLevel.mlmodelc (xcrun coremlcompiler compile).

import CoreML
import Foundation

struct Fixtures: Decodable {
    struct Entry: Decodable {
        let id: String
        let image: String
        let prepared: [String: String]
        let reference: [String: [String: Double]]?
        let truth: [String: String]?
    }
    let release: String
    let entries: [Entry]
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    print("usage: parity-check FIXTURES_DIR COMPILED_MODELS_DIR")
    exit(2)
}
let fixturesDir = URL(fileURLWithPath: arguments[1])
let modelsDir = URL(fileURLWithPath: arguments[2])
let fixtures = try JSONDecoder().decode(
    Fixtures.self, from: Data(contentsOf: fixturesDir.appendingPathComponent("fixtures.json")))

let classifier = try TickClassifier(
    multiheadURL: modelsDir.appendingPathComponent("TickPredictV2.mlmodelc"),
    engorgementURL: modelsDir.appendingPathComponent("EngorgementLevel.mlmodelc"))

var pixelMismatches = 0, pixelBytes = 0, maxPixelDifference = 0
let tasks = ["species", "life_stage", "engorgement_level"]
var labelAgreement = Dictionary(uniqueKeysWithValues: tasks.map { ($0, 0) })
var correct = Dictionary(uniqueKeysWithValues: tasks.map { ($0, 0) })
var maxProbabilityDifference = Dictionary(uniqueKeysWithValues: tasks.map { ($0, 0.0) })
var compared = 0
var slowest = 0.0

for entry in fixtures.entries {
    let image = try ImagePreparation.decode(Data(contentsOf: URL(fileURLWithPath: entry.image)))
    let inputs = try await classifier.preparedInputs(for: image)
    for (task, prepared) in [("multihead", inputs.multihead), ("engorgement_level", inputs.engorgement)] {
        let expected = [UInt8](try Data(contentsOf: URL(fileURLWithPath: entry.prepared[task]!)))
        precondition(expected.count == prepared.pixels.count, "size mismatch for \(entry.id) \(task)")
        for (a, b) in zip(expected, prepared.pixels) where a != b {
            pixelMismatches += 1
            maxPixelDifference = max(maxPixelDifference, abs(Int(a) - Int(b)))
        }
        pixelBytes += expected.count
    }
    guard let reference = entry.reference, let truth = entry.truth else { continue }
    let start = Date()
    let prediction = try await classifier.predict(image: image)
    slowest = max(slowest, Date().timeIntervalSince(start))
    compared += 1
    for (task, result) in [("species", prediction.species!), ("life_stage", prediction.lifeStage),
                           ("engorgement_level", prediction.engorgement)] {
        let expected = reference[task]!
        if result.label == expected.max(by: { $0.value < $1.value })!.key { labelAgreement[task]! += 1 }
        if result.label == truth[task] { correct[task]! += 1 }
        for (label, score) in expected {
            maxProbabilityDifference[task] = max(maxProbabilityDifference[task]!, abs(score - result.scores[label]!))
        }
    }
}

print("Preprocessing: \(fixtures.entries.count) images, \(pixelMismatches) of \(pixelBytes) bytes differ from Python (max difference \(maxPixelDifference))")
for task in tasks {
    print("\(task): \(labelAgreement[task]!)/\(compared) labels match PyTorch, \(correct[task]!)/\(compared) correct, max |dp| \(String(format: "%.4f", maxProbabilityDifference[task]!))")
}
print("Slowest decode + prepare + both models (Mac): \(String(format: "%.0f", slowest * 1000)) ms")
