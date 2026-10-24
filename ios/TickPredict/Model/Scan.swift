import Foundation
import SwiftData

/// A saved scan. Everything stays on the device.
@Model
nonisolated final class Scan {
    var createdAt: Date
    @Attribute(.externalStorage) var imageData: Data
    var thumbnailData: Data?
    var prediction: TickPrediction
    var foundOn: Date
    var bodyArea: BodyArea?
    var place: String
    var notes: String

    init(createdAt: Date, imageData: Data, prediction: TickPrediction) {
        self.createdAt = createdAt
        self.imageData = imageData
        thumbnailData = PhotoLoader.thumbnailJPEG(imageData, maxPixelSize: 240)
        self.prediction = prediction
        foundOn = createdAt
        place = ""
        notes = ""
    }

    var title: String { prediction.title }

    var searchText: String {
        [title, prediction.species.map { Species($0.label).otherName ?? "" } ?? "",
         Engorgement(prediction.engorgement.label).name, place, notes, bodyArea?.name ?? ""].joined(separator: " ")
    }
}

nonisolated enum BodyArea: String, Codable, CaseIterable, Identifiable {
    case head, arm, torso, groin, knee, leg, foot

    var id: String { rawValue }

    var name: String {
        switch self {
        case .head: "Head / neck"
        case .arm: "Arm"
        case .torso: "Torso"
        case .groin: "Groin"
        case .knee: "Back of knee"
        case .leg: "Leg"
        case .foot: "Ankle / foot"
        }
    }

    /// Marker position on the body outline, in its 74 × 150 coordinate space.
    var marker: CGPoint {
        switch self {
        case .head: CGPoint(x: 37, y: 18)
        case .arm: CGPoint(x: 15, y: 52)
        case .torso: CGPoint(x: 37, y: 50)
        case .groin: CGPoint(x: 37, y: 78)
        case .knee: CGPoint(x: 29, y: 104)
        case .leg: CGPoint(x: 45, y: 120)
        case .foot: CGPoint(x: 29, y: 138)
        }
    }
}

extension TickPrediction {
    /// "Blacklegged tick · Nymph · Semi-engorged"; v1.0 scans have no species.
    nonisolated var title: String {
        [species.map { Species($0.label).name }, LifeStage(lifeStage.label).name,
         Engorgement(engorgement.label).name].compactMap { $0 }.joined(separator: " · ")
    }

    nonisolated var confidenceSummary: String {
        [species, lifeStage, engorgement].compactMap { $0?.confidence.percent }.joined(separator: " · ")
            + " model confidence"
    }
}

/// Display names and plain-language explanations for each class.
nonisolated struct LifeStage {
    let label: String
    init(_ label: String) { self.label = label }

    static let order = ["nymph", "adult female", "adult male"]

    var name: String {
        switch label {
        case "nymph": "Nymph"
        case "adult male": "Adult male"
        default: "Adult female"
        }
    }

    var shortName: String {
        switch label {
        case "nymph": "Nymph"
        case "adult male": "Male"
        default: "Female"
        }
    }

    var meaning: String {
        switch label {
        case "nymph":
            "Nymphs are the young stage, about the size of a poppy seed. They're easy to miss, so they often stay attached longer. In deer ticks, they cause most Lyme infections in people."
        case "adult male":
            "Adult males are about the size of a sesame seed and stay small when they feed. Male blacklegged ticks rarely feed long at all."
        default:
            "Adult females are about the size of a sesame seed before feeding and grow much larger as they feed. They're easier to spot than nymphs."
        }
    }
}

nonisolated struct Engorgement {
    let label: String
    init(_ label: String) { self.label = label }

    static let order = ["unengorged", "semi-engorged", "fully engorged"]

    var index: Int { Self.order.firstIndex(of: label) ?? 0 }

    var name: String {
        switch label {
        case "unengorged": "Unengorged"
        case "semi-engorged": "Semi-engorged"
        default: "Fully engorged"
        }
    }

    var shortName: String { ["None", "Semi", "Fully"][index] }

    var meaning: String {
        switch label {
        case "unengorged":
            "Unengorged means it's still flat, with little or no blood taken, so it probably wasn't attached long. It can't tell you exactly how long."
        case "semi-engorged":
            "Semi-engorged means it has taken some blood, so it was likely attached for a while. It can't tell you exactly how long."
        default:
            "Fully engorged means it's swollen with blood, which usually takes days of feeding. Tell your doctor it looked engorged."
        }
    }
}
