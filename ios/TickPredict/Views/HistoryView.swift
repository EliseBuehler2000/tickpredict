import SwiftData
import SwiftUI

struct HistoryView: View {
    @Query(sort: [SortDescriptor(\Scan.foundOn, order: .reverse), SortDescriptor(\Scan.createdAt, order: .reverse)])
    private var scans: [Scan]
    @Environment(\.modelContext) private var context
    @State private var search = ""

    private var filtered: [Scan] {
        let query = search.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { return scans }
        return scans.filter { $0.searchText.localizedStandardContains(query) }
    }

    private var groups: [(title: String, scans: [Scan])] {
        let weekAgo = Calendar.current.date(byAdding: .day, value: -7, to: .now) ?? .now
        // Grouped by when the tick was found, which is the date each row shows.
        let recent = filtered.filter { $0.foundOn >= weekAgo }
        let earlier = filtered.filter { $0.foundOn < weekAgo }
        return [("This week", recent), ("Earlier", earlier)].filter { !$0.1.isEmpty }
    }

    var body: some View {
        Group {
            if scans.isEmpty {
                ContentUnavailableView("No scans yet", systemImage: "clock",
                                       description: Text("Scans you save appear here, with the date and where the bite was."))
            } else if filtered.isEmpty {
                ContentUnavailableView.search(text: search)
            } else {
                List {
                    ForEach(groups, id: \.title) { group in
                        Section {
                            ForEach(group.scans) { scan in
                                NavigationLink(value: Route.saved(scan)) {
                                    ScanRow(scan: scan, showsChevron: false).padding(.horizontal, -14).padding(.vertical, -4)
                                }
                                .listRowBackground(Theme.surface)
                            }
                            .onDelete { offsets in
                                for index in offsets { context.delete(group.scans[index]) }
                                try? context.save()
                            }
                        } header: {
                            Eyebrow(group.title)
                        }
                    }
                }
                .scrollContentBackground(.hidden)
            }
        }
        .background(Theme.ground)
        .navigationTitle("History")
        .searchable(text: $search, prompt: "Search notes or places")
    }
}
