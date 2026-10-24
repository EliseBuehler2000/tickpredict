import CoreML
import Foundation

/// One task's answer: the top label plus every class score.
/// Scores are the models' uncalibrated softmax outputs.
nonisolated struct TaskPrediction: Sendable, Codable, Hashable {
    let label: String
    let scores: [String: Double]

    init(label: String, scores: [String: Double]) {
        self.label = label
        self.scores = scores
    }

    init(scores: [String: Double]) {
        self.init(label: scores.max { $0.value < $1.value }?.key ?? "", scores: scores)
    }

    var confidence: Double { scores[label] ?? 0 }

    /// Classes ordered from most to least likely.
    var ranked: [(label: String, score: Double)] {
        scores.sorted { $0.value > $1.value }.map { ($0.key, $0.value) }
    }
}

nonisolated struct TickPrediction: Sendable, Codable, Hashable {
    let lifeStage: TaskPrediction
    let engorgement: TaskPrediction
    let modelRelease: String
    // SwiftData stores this struct as columns derived from its stored properties.
    // A nested optional struct added after v1.0 reads back as "present but empty"
    // for older scans and traps, so v2.0's species is kept as one optional JSON
    // blob, which reads back as nil.
    private var speciesJSON: Data?

    /// Added in v2.0. Scans saved with v1.0 have no species.
    var species: TaskPrediction? {
        speciesJSON.flatMap { try? JSONDecoder().decode(TaskPrediction.self, from: $0) }
    }

    init(species: TaskPrediction? = nil, lifeStage: TaskPrediction, engorgement: TaskPrediction, modelRelease: String) {
        self.lifeStage = lifeStage
        self.engorgement = engorgement
        self.modelRelease = modelRelease
        speciesJSON = species.flatMap { try? JSONEncoder().encode($0) }
    }
}

nonisolated enum TickClassifierError: Error {
    case missingModel(String)
    case missingMetadata(String)
    case unexpectedOutput(String)
}

/// Runs the bundled models on a photo: the v2 multi-head model for species and
/// life stage, and the v1 engorgement model, which scored better than v2's head.
actor TickClassifier {
    private struct Loaded {
        let model: MLModel
        let spec: PreprocessingSpec
        let metadata: [String: String]
        let release: String
    }

    private let multihead: Loaded
    private let heads: [String: [String]]
    private let engorgement: Loaded
    private let engorgementClasses: [String]

    /// Loads compiled models (`.mlmodelc`) from the given URLs.
    init(multiheadURL: URL, engorgementURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        multihead = try Self.load(multiheadURL, task: "multihead", configuration: configuration)
        engorgement = try Self.load(engorgementURL, task: "engorgement_level", configuration: configuration)
        guard let headsJSON = multihead.metadata["heads"]?.data(using: .utf8),
              let classesJSON = engorgement.metadata["classes"]?.data(using: .utf8)
        else { throw TickClassifierError.missingMetadata("classes") }
        heads = try JSONDecoder().decode([String: [String]].self, from: headsJSON)
        engorgementClasses = try JSONDecoder().decode([String].self, from: classesJSON)
    }

    /// Loads the models bundled with the app.
    init(bundle: Bundle = .main) throws {
        guard let multiheadURL = bundle.url(forResource: "TickPredictV2", withExtension: "mlmodelc")
        else { throw TickClassifierError.missingModel("TickPredictV2") }
        guard let engorgementURL = bundle.url(forResource: "EngorgementLevel", withExtension: "mlmodelc")
        else { throw TickClassifierError.missingModel("EngorgementLevel") }
        try self.init(multiheadURL: multiheadURL, engorgementURL: engorgementURL)
    }

    private static func load(_ url: URL, task: String, configuration: MLModelConfiguration) throws -> Loaded {
        let model = try MLModel(contentsOf: url, configuration: configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard metadata["task"] == task,
              let specJSON = metadata["preprocessing"]?.data(using: .utf8),
              let release = metadata["release"]
        else { throw TickClassifierError.missingMetadata(url.lastPathComponent) }
        return Loaded(model: model, spec: try JSONDecoder().decode(PreprocessingSpec.self, from: specJSON),
                      metadata: metadata, release: release)
    }

    /// Runs both models on encoded image data (JPEG, HEIC, PNG…).
    func predict(imageData: Data) throws -> TickPrediction {
        try predict(image: ImagePreparation.decode(imageData))
    }

    func predict(image: RGBImage) throws -> TickPrediction {
        let inputs = try preparedInputs(for: image)
        let output = try run(multihead, on: inputs.multihead)
        return TickPrediction(species: try head("species", of: output),
                              lifeStage: try head("life_stage", of: output),
                              engorgement: try engorgementPrediction(run(engorgement, on: inputs.engorgement)),
                              modelRelease: multihead.release)
    }

    /// The prepared model inputs, exposed for parity checks. Both models currently
    /// share one recipe, so the photo is prepared once.
    func preparedInputs(for image: RGBImage) throws -> (multihead: RGBImage, engorgement: RGBImage) {
        let first = try ImagePreparation.prepare(image, spec: multihead.spec)
        let second = engorgement.spec == multihead.spec ? first : try ImagePreparation.prepare(image, spec: engorgement.spec)
        return (first, second)
    }

    private func run(_ loaded: Loaded, on prepared: RGBImage) throws -> MLFeatureProvider {
        let input = try ImagePreparation.pixelBuffer(prepared)
        let features = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])
        return try loaded.model.prediction(from: features)
    }

    private func head(_ name: String, of output: MLFeatureProvider) throws -> TaskPrediction {
        guard let classes = heads[name],
              let array = output.featureValue(for: name)?.multiArrayValue,
              array.count == classes.count
        else { throw TickClassifierError.unexpectedOutput(name) }
        var scores: [String: Double] = [:]
        for (index, label) in classes.enumerated() { scores[label] = array[index].doubleValue }
        return TaskPrediction(scores: scores)
    }

    private func engorgementPrediction(_ output: MLFeatureProvider) throws -> TaskPrediction {
        guard let label = output.featureValue(for: "classLabel")?.stringValue,
              let probabilities = output.featureValue(for: "classLabel_probs")?.dictionaryValue as? [String: NSNumber],
              Set(probabilities.keys) == Set(engorgementClasses)
        else { throw TickClassifierError.unexpectedOutput("engorgement_level") }
        return TaskPrediction(label: label, scores: probabilities.mapValues(\.doubleValue))
    }
}
