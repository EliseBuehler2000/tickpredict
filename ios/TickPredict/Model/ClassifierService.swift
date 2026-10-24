import Foundation
import Observation

nonisolated enum ScanError: Error {
    case modelsUnavailable
    case unreadablePhoto
}

/// Loads the models once in the background and runs scans. Each photo is
/// classified once; returning to its screen reuses the same result.
@Observable
final class ClassifierService {
    private var loading: Task<TickClassifier, Error>
    private var runs: [UUID: Task<TickPrediction, Error>] = [:]

    init() {
        loading = Self.load()
    }

    private static func load() -> Task<TickClassifier, Error> {
        Task.detached(priority: .userInitiated) { try TickClassifier() }
    }

    func classify(_ photo: PendingPhoto) async throws -> TickPrediction {
        let run = runs[photo.id] ?? start(photo)
        do {
            return try await run.value
        } catch {
            runs[photo.id] = nil
            if (error as? ScanError) == .modelsUnavailable { loading = Self.load() }
            throw error
        }
    }

    private func start(_ photo: PendingPhoto) -> Task<TickPrediction, Error> {
        let loader = loading
        let data = photo.data
        let run = Task.detached(priority: .userInitiated) { () async throws -> TickPrediction in
            let classifier: TickClassifier
            do { classifier = try await loader.value } catch { throw ScanError.modelsUnavailable }
            do { return try await classifier.predict(imageData: data) } catch { throw ScanError.unreadablePhoto }
        }
        runs[photo.id] = run
        return run
    }
}
