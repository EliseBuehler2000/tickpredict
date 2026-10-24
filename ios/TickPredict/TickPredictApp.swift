import SwiftData
import SwiftUI

@main
struct TickPredictApp: App {
    @State private var classifier = ClassifierService()

    init() {
        let rounded = { (style: UIFont.TextStyle) -> UIFont in
            let base = UIFont.preferredFont(forTextStyle: style)
            let bold = base.fontDescriptor.withSymbolicTraits(.traitBold) ?? base.fontDescriptor
            return UIFont(descriptor: bold.withDesign(.rounded) ?? bold, size: 0)
        }
        UINavigationBar.appearance().largeTitleTextAttributes = [.font: rounded(.largeTitle)]
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(classifier)
        }
        .modelContainer(for: Scan.self)
    }
}

enum AppTab: Hashable { case scan, history, guide }

/// Screens pushed onto any tab's navigation stack.
enum Route: Hashable {
    case newScan(PendingPhoto)
    case saved(Scan)
    case findCare
}

/// A photo waiting to be scanned. Identity is its id, so navigation never hashes the image.
struct PendingPhoto: Hashable {
    let id = UUID()
    let data: Data
    let takenAt: Date

    init(data: Data) {
        self.data = data
        takenAt = min(PhotoLoader.captureDate(data) ?? .now, .now)
    }

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

struct RootView: View {
    @State private var tab = AppTab.scan
    @State private var scanPath: [Route] = []
    @State private var historyPath: [Route] = []
    @State private var guidePath: [Route] = []
    @State private var showCamera = false

    var body: some View {
        TabView(selection: $tab) {
            Tab("Scan", systemImage: "viewfinder", value: .scan) {
                NavigationStack(path: $scanPath) {
                    HomeView(showCamera: $showCamera, openHistory: { tab = .history },
                             openGuide: { tab = .guide }, open: open)
                        .withRoutes(scanAnother: scanAnother)
                }
            }
            Tab("History", systemImage: "clock", value: .history) {
                NavigationStack(path: $historyPath) {
                    HistoryView().withRoutes(scanAnother: scanAnother)
                }
            }
            Tab("Guide", systemImage: "book.pages", value: .guide) {
                NavigationStack(path: $guidePath) {
                    GuideView().withRoutes(scanAnother: scanAnother)
                }
            }
        }
        .tint(Theme.accent)
        .onReceive(NotificationCenter.default.publisher(for: ModelContext.didSave)) { _ in dropDeletedScans() }
        .fullScreenCover(isPresented: $showCamera) {
            CaptureView { data in open(PendingPhoto(data: data)) }
        }
        #if DEBUG
        .task { DebugLaunch.apply(open: open, tab: $tab, path: $scanPath, camera: $showCamera) }
        .modifier(DebugSeed())
        #endif
    }

    private func open(_ photo: PendingPhoto) {
        tab = .scan
        scanPath = [.newScan(photo)]
    }

    /// A scan deleted in one tab may still be open in another; close those screens.
    private func dropDeletedScans() {
        let isDeleted = { (route: Route) -> Bool in
            if case .saved(let scan) = route { return scan.isDeleted || scan.modelContext == nil }
            return false
        }
        if scanPath.contains(where: isDeleted) { scanPath.removeAll(where: isDeleted) }
        if historyPath.contains(where: isDeleted) { historyPath.removeAll(where: isDeleted) }
    }

    private func scanAnother() {
        tab = .scan
        scanPath = []
        showCamera = true
    }
}

extension View {
    func withRoutes(scanAnother: @escaping () -> Void) -> some View {
        navigationDestination(for: Route.self) { route in
            switch route {
            case .newScan(let photo): ResultView(source: .new(photo), scanAnother: scanAnother)
            case .saved(let scan): ResultView(source: .saved(scan), scanAnother: scanAnother)
            case .findCare: FindCareView()
            }
        }
    }
}
