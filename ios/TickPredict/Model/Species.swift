import Foundation

/// The eight species the v2 model knows, with what each can spread to people.
/// Diseases follow CDC's "Where ticks live" page (checked 2026-10-19).
nonisolated struct Species {
    let label: String
    let name: String
    let otherName: String?
    let carries: [String]
    let note: String?
    let spreadsLyme: Bool

    init(_ label: String) {
        self = Self.all[label] ?? Species(label: label, name: label, otherName: nil, carries: [], note: nil, spreadsLyme: false)
    }

    private init(label: String, name: String, otherName: String?, carries: [String], note: String?, spreadsLyme: Bool) {
        self.label = label
        self.name = name
        self.otherName = otherName
        self.carries = carries
        self.note = note
        self.spreadsLyme = spreadsLyme
    }

    static let lymeLabels: Set<String> = ["Ixodes scapularis", "Ixodes pacificus"]

    private static let all: [String: Species] = Dictionary(uniqueKeysWithValues: [
        Species(label: "Ixodes scapularis", name: "Blacklegged tick", otherName: "Deer tick",
                carries: ["Lyme disease", "anaplasmosis", "babesiosis", "ehrlichiosis", "Powassan virus",
                          "hard tick relapsing fever"],
                note: nil, spreadsLyme: true),
        Species(label: "Ixodes pacificus", name: "Western blacklegged tick", otherName: nil,
                carries: ["Lyme disease", "anaplasmosis", "hard tick relapsing fever"],
                note: "Found on the Pacific coast. It looks almost the same as the eastern blacklegged (deer) tick.",
                spreadsLyme: true),
        Species(label: "Dermacentor variabilis", name: "American dog tick", otherName: "Wood tick",
                carries: ["Rocky Mountain spotted fever", "tularemia"],
                note: nil, spreadsLyme: false),
        Species(label: "Amblyomma americanum", name: "Lone star tick", otherName: nil,
                carries: ["ehrlichiosis", "tularemia", "Heartland virus", "Bourbon virus"],
                note: "Its bites are also linked to alpha-gal syndrome, an allergy to red meat, and to STARI, a rash like Lyme's.",
                spreadsLyme: false),
        Species(label: "Amblyomma maculatum", name: "Gulf Coast tick", otherName: nil,
                carries: ["Rickettsia parkeri rickettsiosis, a kind of spotted fever"],
                note: nil, spreadsLyme: false),
        Species(label: "Rhipicephalus sanguineus", name: "Brown dog tick", otherName: nil,
                carries: ["Rocky Mountain spotted fever"],
                note: "Mostly a risk in the Southwest and along the U.S.–Mexico border. It often lives indoors with dogs.",
                spreadsLyme: false),
        Species(label: "Dermacentor andersoni", name: "Rocky Mountain wood tick", otherName: nil,
                carries: ["Rocky Mountain spotted fever", "Colorado tick fever", "tularemia"],
                note: "Found in the Rocky Mountain states. It looks a lot like the American dog tick.",
                spreadsLyme: false),
        Species(label: "Haemaphysalis longicornis", name: "Asian longhorned tick", otherName: nil,
                carries: [],
                note: "Not yet known to make people sick in the U.S. It spreads disease in other countries and can harm livestock.",
                spreadsLyme: false),
    ].map { ($0.label, $0) })

    /// "Lyme disease, anaplasmosis and babesiosis" style list.
    var carriesText: String {
        carries.formatted(.list(type: .and))
    }
}

/// How the species, life stage and engorgement combine for Lyme disease.
/// Doctors weigh a single preventive antibiotic dose when a blacklegged tick was
/// attached about 36 hours or more and it's within 72 hours of removal (IDSA 2020).
nonisolated enum LymeRisk {
    case higher, possible, lower, maleTick, notLymeTick

    /// Chance the model gives to either blacklegged species.
    static func blackleggedScore(_ prediction: TickPrediction) -> Double? {
        prediction.species.map { species in Species.lymeLabels.reduce(0) { $0 + (species.scores[$1] ?? 0) } }
    }

    /// Engorged blacklegged ticks are under-represented in the species training
    /// photos: on Penn's development photos (all blacklegged), only 6 of 16 fully
    /// engorged ones were named correctly. So an engorged tick is never ruled out.
    static func isEngorged(_ prediction: TickPrediction) -> Bool {
        prediction.engorgement.label != "unengorged"
    }

    /// Scans saved with v1.0 have no species, so there is no Lyme reading.
    init?(_ prediction: TickPrediction) {
        guard let blacklegged = Self.blackleggedScore(prediction) else { return nil }
        if blacklegged < 0.2 && !Self.isEngorged(prediction) {
            self = .notLymeTick
        } else if prediction.lifeStage.label == "adult male" {
            self = .maleTick
        } else {
            switch prediction.engorgement.label {
            case "fully engorged": self = .higher
            case "semi-engorged": self = .possible
            default: self = .lower
            }
        }
    }

    var title: String {
        switch self {
        case .higher: "Higher Lyme risk"
        case .possible: "Possible Lyme risk"
        case .lower: "Lower Lyme risk"
        case .maleTick: "Low Lyme risk"
        case .notLymeTick: "Doesn't spread Lyme"
        }
    }

    var isUrgent: Bool { self == .higher || self == .possible }

    var explanation: String {
        switch self {
        case .higher:
            "A fully engorged blacklegged tick was likely attached 36 hours or more, long enough to spread Lyme. If it came off within the last 72 hours, a doctor may offer a single preventive antibiotic dose. Call a doctor today."
        case .possible:
            "A semi-engorged blacklegged tick may have been attached long enough to spread Lyme. If it came off within the last 72 hours, ask a doctor whether a preventive antibiotic dose makes sense."
        case .lower:
            "An unengorged tick likely wasn't attached long, and Lyme usually takes 36 hours or more to spread. Watch the bite for 30 days."
        case .maleTick:
            "Adult male blacklegged ticks rarely feed long enough to spread Lyme. Watch the bite for 30 days."
        case .notLymeTick:
            "This kind of tick isn't known to spread Lyme disease. Still watch the bite for 30 days."
        }
    }
}
