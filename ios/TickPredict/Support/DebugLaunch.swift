#if DEBUG
import SwiftData
import SwiftUI

/// Debug-only launch arguments for reviewing screens in the Simulator:
///   -demoImage /path/photo.jpg     open a new scan of that photo (relative: in Documents)
///   -tab history|guide             start on a tab
///   -findCare                      push Find care on the Scan tab
///   -camera                        open the camera
///   -resetData YES                 delete all saved scans first (UI tests)
///   -seedScans /a.jpg,/b.jpg       save scans of those photos (if history is empty)
enum DebugLaunch {
    static func apply(open: (PendingPhoto) -> Void, tab: Binding<AppTab>, path: Binding<[Route]>, camera: Binding<Bool>) {
        let defaults = UserDefaults.standard
        if let image = defaults.string(forKey: "demoImage"),
           let data = FileManager.default.contents(atPath: resolve(image)) {
            open(PendingPhoto(data: data))
        }
        switch defaults.string(forKey: "tab") {
        case "history": tab.wrappedValue = .history
        case "guide": tab.wrappedValue = .guide
        default: break
        }
        if defaults.bool(forKey: "findCare") { path.wrappedValue.append(.findCare) }
        if defaults.bool(forKey: "camera") { camera.wrappedValue = true }
    }

    /// Relative paths are inside the app's Documents folder, for photos copied onto a device.
    static func resolve(_ path: String) -> String {
        path.hasPrefix("/") ? path : URL.documentsDirectory.appendingPathComponent(path).path
    }

    static func seed(context: ModelContext, classifier: ClassifierService) async {
        if UserDefaults.standard.bool(forKey: "resetData") {
            try? context.delete(model: Scan.self)
            try? context.save()
        }
        guard let list = UserDefaults.standard.string(forKey: "seedScans"),
              (try? context.fetchCount(FetchDescriptor<Scan>())) == 0 else { return }
        let details: [(BodyArea, String, String, Int)] = [
            (.foot, "Backyard", "Probably from mowing on Saturday", 0),
            (.knee, "Trail", "", 6),
            (.torso, "Campsite", "Saw it two days after the trip", 21),
            (.arm, "Garden", "", 40),
        ]
        for (index, path) in list.split(separator: ",").enumerated() {
            guard let data = FileManager.default.contents(atPath: String(path)),
                  let prediction = try? await classifier.classify(PendingPhoto(data: data)) else { continue }
            let (area, place, notes, daysAgo) = details[index % details.count]
            let date = Calendar.current.date(byAdding: .day, value: -daysAgo, to: .now)!
            let scan = Scan(createdAt: date, imageData: data, prediction: prediction)
            scan.bodyArea = area
            scan.place = place
            scan.notes = notes
            context.insert(scan)
        }
        try? context.save()
    }
}

struct DebugSeed: ViewModifier {
    @Environment(\.modelContext) private var context
    @Environment(ClassifierService.self) private var classifier

    func body(content: Content) -> some View {
        content.task { await DebugLaunch.seed(context: context, classifier: classifier) }
    }
}
#endif
