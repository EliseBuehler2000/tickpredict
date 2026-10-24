import SwiftData
import SwiftUI

/// Saves a new scan, or edits a saved one's notes.
struct SaveScanView: View {
    let prediction: TickPrediction
    let imageData: Data
    let takenAt: Date
    let editing: Scan?
    let onSave: (Scan) -> Void

    private enum Field { case place, notes }

    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @FocusState private var focus: Field?
    @State private var foundOn = Date()
    @State private var bodyArea: BodyArea?
    @State private var place = ""
    @State private var notes = ""
    @State private var thumbnail: Data?
    @State private var original: Snapshot?
    @State private var isSaving = false
    @State private var saveFailed = false
    @State private var confirmDiscard = false

    /// Field values, to tell whether anything changed before discarding.
    private struct Snapshot: Equatable {
        var foundOn: Date
        var bodyArea: BodyArea?
        var place: String
        var notes: String
    }

    private var current: Snapshot { Snapshot(foundOn: foundOn, bodyArea: bodyArea, place: place, notes: notes) }
    private var hasChanges: Bool { original != nil && current != original }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    summary

                    section("When") {
                        DatePicker("Found", selection: $foundOn, in: ...max(foundOn, .now), displayedComponents: .date)
                            .padding(.horizontal, 14)
                            .frame(minHeight: 50)
                            .background(Theme.surface, in: .rect(cornerRadius: 16))
                    }

                    section("Where on the body") {
                        HStack(alignment: .center, spacing: 14) {
                            BodyOutline(marker: bodyArea?.marker)
                            FlowLayout(spacing: 8) {
                                ForEach(BodyArea.allCases) { area in
                                    Chip(title: area.name, selected: bodyArea == area) {
                                        bodyArea = bodyArea == area ? nil : area
                                    }
                                }
                            }
                        }
                        .padding(14)
                        .background(Theme.surface, in: .rect(cornerRadius: 16))
                        .sensoryFeedback(.selection, trigger: bodyArea)
                    }

                    section("Where were you") {
                        TextField("e.g. Backyard, hiking trail", text: $place)
                            .focused($focus, equals: .place)
                            .submitLabel(.next)
                            .onSubmit { focus = .notes }
                            .padding(.horizontal, 14)
                            .frame(minHeight: 50)
                            .background(Theme.surface, in: .rect(cornerRadius: 16))
                    }

                    section("Notes") {
                        TextField("Anything to tell your doctor", text: $notes, axis: .vertical)
                            .focused($focus, equals: .notes)
                            .lineLimit(3...8)
                            .padding(14)
                            .background(Theme.surface, in: .rect(cornerRadius: 16))
                    }

                    Text("Scans are stored only on this iPhone.")
                        .font(.caption)
                        .foregroundStyle(Theme.muted)
                        .frame(maxWidth: .infinity)
                }
                .padding(16)
            }
            .scrollDismissesKeyboard(.interactively)
            .background(Theme.ground)
            .navigationTitle(editing == nil ? "Save scan" : "Edit scan")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", role: .cancel) {
                        if hasChanges { confirmDiscard = true } else { dismiss() }
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(editing == nil ? "Save" : "Done", role: .confirm, action: save)
                        .disabled(isSaving)
                }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Hide keyboard", systemImage: "keyboard.chevron.compact.down") { focus = nil }
                }
            }
            .confirmationDialog("Discard your changes?", isPresented: $confirmDiscard, titleVisibility: .visible) {
                Button("Discard changes", role: .destructive) { dismiss() }
            }
            .alert("Couldn't save the scan", isPresented: $saveFailed) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("Your iPhone may be low on storage. Free up some space and try again.")
            }
        }
        .interactiveDismissDisabled(hasChanges)
        .onAppear(perform: fill)
        .task {
            let data = imageData
            thumbnail = await Task.detached { PhotoLoader.thumbnailJPEG(data, maxPixelSize: 160) }.value
        }
    }

    private var summary: some View {
        HStack(spacing: 12) {
            ScanThumbnail(data: thumbnail, size: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text(prediction.title)
                    .font(.body.weight(.semibold))
                Text(prediction.confidenceSummary)
                    .font(.footnote)
                    .foregroundStyle(Theme.muted)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: .rect(cornerRadius: 16))
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Eyebrow(title).padding(.leading, 4)
            content()
        }
    }

    private func fill() {
        guard original == nil else { return }
        if let editing {
            foundOn = editing.foundOn
            bodyArea = editing.bodyArea
            place = editing.place
            notes = editing.notes
        } else {
            foundOn = takenAt
        }
        original = current
    }

    private func save() {
        guard !isSaving else { return }
        isSaving = true
        let scan = editing ?? Scan(createdAt: takenAt, imageData: imageData, prediction: prediction)
        scan.foundOn = foundOn
        scan.bodyArea = bodyArea
        scan.place = place.trimmingCharacters(in: .whitespacesAndNewlines)
        scan.notes = notes.trimmingCharacters(in: .whitespacesAndNewlines)
        if editing == nil { context.insert(scan) }
        do {
            try context.save()
        } catch {
            if editing == nil { context.delete(scan) } else { context.rollback() }
            isSaving = false
            saveFailed = true
            return
        }
        onSave(scan)
        dismiss()
    }
}

struct Chip: View {
    let title: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(selected ? .semibold : .regular))
                .foregroundStyle(selected ? Theme.accent : Theme.body)
                .padding(.horizontal, 12)
                .frame(minHeight: 36)
                .background(selected ? Theme.accentTint : Theme.surface, in: .capsule)
                .overlay(Capsule().strokeBorder(selected ? Theme.accent : Theme.line, lineWidth: 1.5))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// Simple front-facing figure with the bite spot marked.
struct BodyOutline: View {
    let marker: CGPoint?

    var body: some View {
        Canvas { context, size in
            let shade = GraphicsContext.Shading.color(Theme.track)
            context.fill(Path(ellipseIn: CGRect(x: 26, y: 3, width: 22, height: 22)), with: shade)
            for rect in [CGRect(x: 22, y: 28, width: 30, height: 50), CGRect(x: 10, y: 30, width: 10, height: 46),
                         CGRect(x: 54, y: 30, width: 10, height: 46), CGRect(x: 23, y: 74, width: 12, height: 70),
                         CGRect(x: 39, y: 74, width: 12, height: 70)] {
                context.fill(Path(roundedRect: rect, cornerRadius: min(rect.width, rect.height) / 2), with: shade)
            }
            if let marker {
                let dot = CGRect(x: marker.x - 7, y: marker.y - 7, width: 14, height: 14)
                context.fill(Path(ellipseIn: dot), with: .color(Theme.bite))
                context.stroke(Path(ellipseIn: dot), with: .color(Theme.surface), lineWidth: 2)
            }
        }
        .frame(width: 74, height: 150)
        .animation(.snappy, value: marker)
        .accessibilityHidden(true)
    }
}

/// Wraps children onto new lines as needed.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    var centered = false

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(proposal.width ?? .infinity, subviews)
        return CGSize(width: proposal.width ?? rows.width, height: rows.height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for (index, point) in arrange(bounds.width, subviews).points.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + point.x, y: bounds.minY + point.y),
                                  proposal: ProposedViewSize(width: min(subviews[index].sizeThatFits(.unspecified).width, bounds.width), height: nil))
        }
    }

    private func arrange(_ width: CGFloat, _ subviews: Subviews) -> (points: [CGPoint], width: CGFloat, height: CGFloat) {
        var points: [CGPoint] = []
        var rowStart = 0
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0, maxX: CGFloat = 0
        func finishRow(upTo end: Int) {
            guard centered, width.isFinite, end > rowStart else { return }
            let shift = max(0, (width - (x - spacing)) / 2)
            for i in rowStart ..< end { points[i].x += shift }
        }
        for (index, subview) in subviews.enumerated() {
            let size = subview.sizeThatFits(.unspecified)
            let itemWidth = min(size.width, width)
            if x > 0 && x + itemWidth > width {
                finishRow(upTo: index)
                rowStart = index
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            points.append(CGPoint(x: x, y: y))
            x += itemWidth + spacing
            maxX = max(maxX, x - spacing)
            rowHeight = max(rowHeight, subview.sizeThatFits(ProposedViewSize(width: itemWidth, height: nil)).height)
        }
        finishRow(upTo: subviews.count)
        return (points, maxX, y + rowHeight)
    }
}
