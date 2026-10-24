import SwiftData
import SwiftUI

struct ResultView: View {
    enum Source { case new(PendingPhoto), saved(Scan) }

    let source: Source
    let scanAnother: () -> Void

    @Environment(ClassifierService.self) private var classifier
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var typeSize
    @State private var prediction: TickPrediction?
    @State private var failure: ScanError?
    @State private var stepsDone = 0
    @State private var savedScan: Scan?
    @State private var showSave = false
    @State private var showDetails = false
    @State private var confirmDelete = false
    @State private var deleting = false
    @State private var photo: UIImage?
    @State private var attempt = 0
    @State private var saves = 0
    @State private var storeChanges = 0

    init(source: Source, scanAnother: @escaping () -> Void) {
        self.source = source
        self.scanAnother = scanAnother
        if case .saved(let scan) = source {
            _savedScan = State(initialValue: scan)
            _prediction = State(initialValue: scan.prediction)
        }
    }

    private var scan: Scan? { savedScan }

    /// A saved scan that was deleted, here or in another tab. A deleted model must not be read.
    /// (A new scan whose saved copy is deleted just becomes unsaved again; see `storeDidSave`.)
    private var isGone: Bool {
        _ = storeChanges
        if deleting { return true }
        guard !isNew, let savedScan else { return false }
        return savedScan.isDeleted || savedScan.modelContext == nil
    }

    private var isNew: Bool {
        if case .new = source { true } else { false }
    }

    private var imageData: Data {
        switch source {
        case .new(let pending): pending.data
        case .saved(let scan): scan.imageData
        }
    }

    private var takenAt: Date {
        switch source {
        case .new(let pending): pending.takenAt
        case .saved(let scan): scan.createdAt
        }
    }

    var body: some View {
        Group {
            if isGone {
                ContentUnavailableView("Scan deleted", systemImage: "trash",
                                       description: Text("This scan is no longer in your history."))
            } else {
                content
            }
        }
        .background(Theme.ground)
        .navigationTitle(isNew && scan == nil ? "Result" : "Scan")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { if !isGone { toolbar } }
        .sheet(isPresented: $showSave) {
            if let prediction, !isGone {
                SaveScanView(prediction: prediction, imageData: imageData, takenAt: takenAt, editing: scan) {
                    savedScan = $0
                    saves += 1
                }
            }
        }
        .confirmationDialog("Delete this scan?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete scan", role: .destructive, action: delete)
        } message: {
            Text("The photo and your notes will be removed from this iPhone.")
        }
        .sensoryFeedback(.success, trigger: prediction != nil)
        .sensoryFeedback(.success, trigger: saves)
        .onReceive(NotificationCenter.default.publisher(for: ModelContext.didSave)) { _ in storeDidSave() }
        .task(id: attempt) { await load() }
    }

    private var content: some View {
        ScrollView {
            VStack(spacing: 14) {
                photoHeader
                if let prediction {
                    results(prediction)
                } else if let failure {
                    failureCard(failure)
                } else {
                    analyzing
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 32)
        }
    }

    private func load() async {
        guard !isGone else { return }
        if photo == nil {
            let data = imageData
            photo = await Task.detached { PhotoLoader.image(data, maxPixelSize: 1400) }.value
        }
        guard prediction == nil, case .new(let pending) = source else { return }
        failure = nil
        stepsDone = 0
        // The models start straight away; the steps are revealed over about five
        // seconds to show what is being checked. A slow first run keeps the last
        // step spinning, and a failure shows at once.
        let classifier = classifier
        let durations = AnalysisSteps.durations()
        let outcome = await withTaskGroup(of: Result<TickPrediction, ScanError>?.self) { group in
            group.addTask {
                do { return .success(try await classifier.classify(pending)) }
                catch { return .failure((error as? ScanError) ?? .unreadablePhoto) }
            }
            group.addTask {
                for (index, seconds) in durations.dropLast().enumerated() {
                    try? await Task.sleep(for: .seconds(seconds))
                    if Task.isCancelled { return nil }
                    await MainActor.run { withAnimation(.snappy) { stepsDone = index + 1 } }
                }
                return nil
            }
            var outcome: Result<TickPrediction, ScanError>?
            for await value in group {
                guard let value else { continue }
                outcome = value
                if case .failure = value { group.cancelAll() }
            }
            return outcome
        }
        guard !Task.isCancelled, let outcome else { return }
        switch outcome {
        case .success(let result):
            try? await Task.sleep(for: .seconds(durations.last ?? 0))
            guard !Task.isCancelled else { return }
            withAnimation(.snappy) { stepsDone = AnalysisSteps.steps.count }
            try? await Task.sleep(for: .milliseconds(250))
            withAnimation(.smooth) { prediction = result }
            AccessibilityNotification.Announcement("Result: \(result.title)").post()
        case .failure(let error):
            failure = error
        }
    }

    private func storeDidSave() {
        if isNew, let savedScan, savedScan.isDeleted || savedScan.modelContext == nil {
            self.savedScan = nil
        }
        storeChanges += 1
    }

    private func delete() {
        guard let scan else { return }
        deleting = true
        context.delete(scan)
        try? context.save()
        dismiss()
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        if prediction != nil {
            ToolbarItem(placement: .topBarTrailing) {
                if scan == nil {
                    Button("Save") { showSave = true }.fontWeight(.semibold)
                } else {
                    Menu {
                        Button("Edit details", systemImage: "square.and.pencil") { showSave = true }
                        Button("Delete scan", systemImage: "trash", role: .destructive) { confirmDelete = true }
                    } label: {
                        Image(systemName: "ellipsis")
                    }
                    .accessibilityLabel("Scan options")
                }
            }
        }
    }

    private func failureCard(_ failure: ScanError) -> some View {
        Card {
            Label(failure == .modelsUnavailable ? "Couldn't start the models" : "Couldn't read this photo",
                  systemImage: "exclamationmark.triangle")
                .font(.headline)
            Text(failure == .modelsUnavailable
                 ? "TickPredict's models didn't load. Try again, or close and reopen the app."
                 : "This file couldn't be opened as a photo. Try another photo of the tick.")
                .font(.subheadline)
                .foregroundStyle(Theme.body)
            HStack(spacing: 10) {
                Button("Try again") { attempt += 1 }
                    .buttonStyle(.glass)
                Button("New photo", action: scanAnother)
                    .buttonStyle(.glassProminent)
            }
            .font(.headline)
        }
    }

    // MARK: - Sections

    private var photoHeader: some View {
        Theme.photoBackdrop
            .overlay {
                if let photo { Image(uiImage: photo).resizable().scaledToFill() }
            }
            .overlay(alignment: .bottomLeading) {
                Text(Calendar.current.isDateInToday(takenAt)
                     ? "Today, " + takenAt.formatted(date: .omitted, time: .shortened)
                     : takenAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.footnote.weight(.medium))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .glassEffect(.regular, in: .capsule)
                    .padding(12)
            }
        .frame(height: 220)
        .clipShape(.rect(cornerRadius: 22))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Tick photo, \(takenAt.formatted(date: .abbreviated, time: .shortened))")
    }

    private var analyzing: some View {
        AnalysisSteps(done: stepsDone)
    }

    @ViewBuilder private func results(_ prediction: TickPrediction) -> some View {
        let stage = LifeStage(prediction.lifeStage.label)
        let engorgement = Engorgement(prediction.engorgement.label)

        if let species = prediction.species { speciesCard(species, engorgement: engorgement) }

        let cards = typeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(spacing: 10))
            : AnyLayout(HStackLayout(alignment: .top, spacing: 10))
        cards {
            AnswerCard(title: "Life stage", value: stage.name, confidence: prediction.lifeStage.confidence) {
                SegmentScale(labels: LifeStage.order.map { LifeStage($0).shortName },
                             colors: LifeStage.order.map { _ in Theme.accent },
                             selected: LifeStage.order.firstIndex(of: stage.label) ?? 0)
            }
            AnswerCard(title: "Engorgement", value: engorgement.name, confidence: prediction.engorgement.confidence) {
                SegmentScale(labels: Engorgement.order.map { Engorgement($0).shortName },
                             colors: Theme.engorgementRamp,
                             selected: engorgement.index,
                             ramp: true)
            }
        }

        if let risk = LymeRisk(prediction) { lymeCard(risk, prediction: prediction) }

        let engorgementMeaning = stage.label == "adult male"
            ? "Males stay small when they feed, so engorgement says little about how long it was attached."
            : engorgement.meaning

        Card {
            Text("What this means").font(.rounded(.title3))
            Text("**\(stage.name).** \(stage.meaning)")
            Text("**\(engorgement.name).** \(engorgementMeaning)")
        }
        .font(.subheadline)
        .foregroundStyle(Theme.body)

        if let scan { ScanNotesCard(scan: scan) }

        nextSteps

        details(prediction)

        if isNew {
            VStack(spacing: 10) {
                if scan == nil {
                    Button { showSave = true } label: {
                        Text("Save to history").frame(maxWidth: .infinity, minHeight: 36)
                    }
                    .buttonStyle(.glassProminent)
                } else {
                    Label("Saved to History", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(Theme.accent)
                        .frame(maxWidth: .infinity, minHeight: 44)
                }
                Button(action: scanAnother) {
                    Text("Scan another").frame(maxWidth: .infinity, minHeight: 36)
                }
                .buttonStyle(.glass)
            }
            .font(.headline)
            .padding(.top, 4)
        }

        Text(prediction.species == nil
             ? "Scanned with an earlier version that didn't identify species. Informational, not a diagnosis."
             : "Informational, not a diagnosis.")
            .font(.caption)
            .foregroundStyle(Theme.muted)
            .multilineTextAlignment(.center)
            .padding(.top, 4)
    }

    private func speciesCard(_ task: TaskPrediction, engorgement: Engorgement) -> some View {
        let species = Species(task.label)
        let runnerUp = task.ranked.dropFirst().first
        let carries = species.carriesText
        return VStack(alignment: .leading, spacing: 10) {
            Eyebrow("Species")
            HStack(alignment: .firstTextBaseline) {
                Text(species.name)
                    .font(.rounded(.title2))
                    .foregroundStyle(Theme.ink)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                Text(task.confidence.percent).font(.subheadline.weight(.bold)).foregroundStyle(Theme.accent)
            }
            Text("\(species.otherName.map { "\($0) · " } ?? "")\(Text(species.label).italic())")
                .font(.subheadline)
                .foregroundStyle(Theme.muted)
            ConfidenceBar(value: task.confidence)
            if engorgement.label == "fully engorged" {
                Text(species.spreadsLyme
                     ? "Fully engorged ticks are harder to identify."
                     : "Fully engorged ticks are harder to identify. It could still be a blacklegged tick.")
                    .font(.footnote)
                    .foregroundStyle(Theme.muted)
            } else if task.confidence < 0.6, let runnerUp {
                Text("Could also be a \(Species(runnerUp.label).name.lowercased()) (\(runnerUp.score.percent)).")
                    .font(.footnote)
                    .foregroundStyle(Theme.muted)
            }
            Divider().overlay(Theme.line)
            VStack(alignment: .leading, spacing: 4) {
                Text("Can carry").font(.caption.weight(.semibold)).foregroundStyle(Theme.muted)
                Text(carries.isEmpty ? "No illness in people confirmed in the U.S." : carries.prefix(1).uppercased() + carries.dropFirst())
                    .font(.subheadline)
                    .foregroundStyle(Theme.ink)
                if let note = species.note {
                    Text(note).font(.footnote).foregroundStyle(Theme.body)
                }
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: .rect(cornerRadius: 20))
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("speciesCard")
    }

    private func lymeCard(_ risk: LymeRisk, prediction: TickPrediction) -> some View {
        let blacklegged = LymeRisk.blackleggedScore(prediction) ?? 0
        return Card(fill: risk.isUrgent ? Theme.warningFill : Theme.accentTint) {
            Label {
                Text(risk.title).font(.rounded(.title3))
            } icon: {
                Image(systemName: risk.isUrgent ? "exclamationmark.circle.fill" : "checkmark.shield")
            }
            .foregroundStyle(risk.isUrgent ? Theme.warningInk : Theme.ink)
            Text(risk.explanation)
                .font(.subheadline)
                .foregroundStyle(Theme.body)
            if risk != .notLymeTick, blacklegged < 0.8 {
                Text(LymeRisk.isEngorged(prediction)
                     ? "Engorged ticks are hard to identify from a photo, so this treats it as a possible blacklegged tick."
                     : "It may not be a blacklegged tick (\(blacklegged.percent) likely), but this treats it as one to be safe.")
                    .font(.footnote)
                    .foregroundStyle(Theme.body)
            }
            if risk.isUrgent {
                NavigationLink(value: Route.findCare) {
                    Label("Find care nearby", systemImage: "mappin.and.ellipse")
                        .font(.headline)
                        .frame(maxWidth: .infinity, minHeight: 36)
                }
                .buttonStyle(.glassProminent)
            }
        }
    }

    private var nextSteps: some View {
        Card(fill: Theme.accentTint) {
            Text("What to do next").font(.rounded(.title3))
            VStack(alignment: .leading, spacing: 14) {
                StepRow(icon: "archivebox", title: "Keep the tick",
                        text: "Seal it in a bag or tape it to paper. Your doctor may want to see it.")
                StepRow(icon: "calendar", title: "Note the date and bite spot",
                        text: "Save this scan and TickPredict keeps it for you.")
                StepRow(icon: "eye", title: "Watch for 30 days",
                        text: "An expanding rash, fever, chills, or aches are reasons to call.")
                StepRow(icon: "stethoscope", title: "Talk to a doctor",
                        text: "Especially for an engorged tick. Show them this result.")
            }
            NavigationLink(value: Route.findCare) {
                Label("Find care nearby", systemImage: "mappin.and.ellipse")
                    .font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 36)
            }
            .buttonStyle(.glassProminent)
            .padding(.top, 2)
        }
    }

    private func details(_ prediction: TickPrediction) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.snappy) { showDetails.toggle() }
            } label: {
                HStack {
                    Text("Details").font(.body.weight(.semibold)).foregroundStyle(Theme.ink)
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Theme.muted)
                        .rotationEffect(.degrees(showDetails ? 180 : 0))
                }
                .frame(minHeight: 56)
                .padding(.horizontal, 18)
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .accessibilityValue(showDetails ? "Expanded" : "Collapsed")

            if showDetails {
                Divider().overlay(Theme.line)
                VStack(alignment: .leading, spacing: 18) {
                    if let species = prediction.species {
                        ScoreTable(title: "Species scores", task: species) { Species($0).name }
                    }
                    ScoreTable(title: "Life stage scores", task: prediction.lifeStage) { LifeStage($0).name }
                    ScoreTable(title: "Engorgement scores", task: prediction.engorgement) { Engorgement($0).name }
                    VStack(alignment: .leading, spacing: 6) {
                        Eyebrow("Models · \(prediction.modelRelease)")
                        if prediction.species == nil {
                            Text("Life stage: ConvNeXt Tiny, 320 px. 107 of 110 research photos correct.")
                        } else {
                            Text("Species and life stage: ConvNeXt Tiny, 384 px, averaged over 8 rotated views. On 1,304 held-out iNaturalist photos, species was right 85% of the time, and blacklegged vs. other ticks 97%. Life stage: 106 of 110 research photos correct.")
                        }
                        Text("Engorgement: EfficientNet V2 S, 384 px, averaged over 8 rotated views (v1.0). 95 of 110 research photos correct.")
                        Text("Scores are uncalibrated model confidence, not the chance the answer is right. Runs entirely on this iPhone.")
                    }
                    .font(.footnote)
                    .foregroundStyle(Theme.body)
                }
                .padding(18)
                .transition(.opacity)
            }
        }
        .background(Theme.surface, in: .rect(cornerRadius: 20))
    }
}

// MARK: - Pieces

/// What the models check, revealed one step at a time while a photo is analyzed.
struct AnalysisSteps: View {
    let done: Int

    static let steps: [(title: String, detail: String)] = [
        ("Preparing the photo", "Straightening, trimming, and resizing the photo"),
        ("Identifying the species", "Using a trained ConvNeXt Tiny model"),
        ("Checking the life stage", "Using a trained ConvNeXt Tiny model"),
        ("Measuring engorgement", "Using a trained EfficientNet V2 S model"),
        ("Weighing Lyme risk", "Combining all three results"),
    ]

    /// About five seconds in all, split unevenly so it doesn't feel mechanical.
    static func durations() -> [Double] {
        let weights = steps.indices.map { _ in Double.random(in: 0.6...1.4) }
        let total = Double.random(in: 4.5...5.5)
        return weights.map { $0 / weights.reduce(0, +) * total }
    }

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 2) {
                Text("Analyzing").font(.rounded(.title3))
                Text("On your iPhone. The photo never leaves it.")
                    .font(.subheadline)
                    .foregroundStyle(Theme.muted)
            }
            VStack(alignment: .leading, spacing: 14) {
                ForEach(Self.steps.indices, id: \.self) { index in
                    row(index)
                }
            }
            .padding(.top, 4)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Analyzing. \(Self.steps[min(done, Self.steps.count - 1)].title).")
    }

    private func row(_ index: Int) -> some View {
        let step = Self.steps[index]
        let isDone = index < done
        let isActive = index == done
        return HStack(alignment: .top, spacing: 12) {
            ZStack {
                if isDone {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Theme.accent)
                        .transition(.scale.combined(with: .opacity))
                } else if isActive {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "circle").foregroundStyle(Theme.track)
                }
            }
            .font(.system(size: 20))
            .frame(width: 24, height: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(step.title)
                    .font(.body.weight(isActive ? .semibold : .regular))
                    .foregroundStyle(isDone || isActive ? Theme.ink : Theme.muted)
                if isDone || isActive {
                    Text(step.detail)
                        .font(.footnote)
                        .foregroundStyle(Theme.muted)
                        .transition(.opacity)
                }
            }
            Spacer(minLength: 0)
        }
    }
}

struct AnswerCard<Scale: View>: View {
    let title: String
    let value: String
    let confidence: Double
    @ViewBuilder var scale: Scale

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Eyebrow(title)
            Text(value)
                .font(.rounded(.title2))
                .foregroundStyle(Theme.ink)
                .minimumScaleFactor(0.75)
                .fixedSize(horizontal: false, vertical: true)
                .frame(minHeight: 56, alignment: .topLeading)
            scale
            Divider().overlay(Theme.line)
            HStack(alignment: .firstTextBaseline) {
                Text("Confidence").font(.caption).foregroundStyle(Theme.muted)
                Spacer()
                Text(confidence.percent).font(.subheadline.weight(.bold)).foregroundStyle(Theme.accent)
            }
            ConfidenceBar(value: confidence)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: .rect(cornerRadius: 20))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title): \(value), model confidence \(confidence.percent)")
    }
}

/// Segmented scale with the predicted step highlighted.
struct SegmentScale: View {
    let labels: [String]
    let colors: [Color]
    let selected: Int
    /// Every step keeps its own colour and the selected one is outlined,
    /// instead of only the selected step being filled.
    var ramp = false

    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                ForEach(labels.indices, id: \.self) { index in
                    Capsule()
                        .fill(fill(index))
                        .frame(height: 6)
                        .overlay {
                            if index == selected && ramp {
                                Capsule().strokeBorder(Theme.accent, lineWidth: 1.5).padding(-3)
                            }
                        }
                }
            }
            HStack(spacing: 4) {
                ForEach(labels.indices, id: \.self) { index in
                    Text(labels[index])
                        .font(.caption2.weight(index == selected ? .bold : .regular))
                        .foregroundStyle(index == selected ? Theme.ink : Theme.muted)
                        .frame(maxWidth: .infinity,
                               alignment: index == 0 ? .leading : index == labels.count - 1 ? .trailing : .center)
                }
            }
        }
    }

    private func fill(_ index: Int) -> Color {
        if ramp { return colors[index] }
        return index == selected ? colors[index] : Theme.track
    }
}

struct StepRow: View {
    let icon: String
    let title: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 19))
                .foregroundStyle(Theme.accent)
                .frame(width: 26)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.body.weight(.semibold)).foregroundStyle(Theme.ink)
                Text(text).font(.subheadline).foregroundStyle(Theme.body)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

struct ScoreTable: View {
    let title: String
    let task: TaskPrediction
    let name: (String) -> String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Eyebrow(title)
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 8) {
                ForEach(task.ranked, id: \.label) { row in
                    GridRow {
                        Text(name(row.label))
                        ConfidenceBar(value: row.score,
                                      tint: row.label == task.label ? Theme.accent : Theme.muted.opacity(0.6))
                            .frame(minWidth: 40)
                        Text(row.score.formatted(.number.precision(.fractionLength(2))))
                            .monospacedDigit()
                            .gridColumnAlignment(.trailing)
                    }
                    .font(.subheadline)
                    .foregroundStyle(Theme.ink)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("\(name(row.label)): \(row.score.percent)")
                }
            }
        }
    }
}

struct ScanNotesCard: View {
    let scan: Scan

    var body: some View {
        Card {
            Text("Your notes").font(.rounded(.title3))
            LabeledContent("Found", value: scan.foundOn.formatted(date: .abbreviated, time: .omitted))
            if let area = scan.bodyArea { LabeledContent("Bite spot", value: area.name) }
            if !scan.place.isEmpty { LabeledContent("Where", value: scan.place) }
            if !scan.notes.isEmpty {
                Text(scan.notes).foregroundStyle(Theme.body)
            }
        }
        .font(.subheadline)
    }
}
