import PhotosUI
import SwiftData
import SwiftUI

struct HomeView: View {
    @Binding var showCamera: Bool
    let openHistory: () -> Void
    let openGuide: () -> Void
    let open: (PendingPhoto) -> Void

    @Query(sort: \Scan.createdAt, order: .reverse) private var scans: [Scan]
    @State private var pickerItem: PhotosPickerItem?
    @State private var loadingPhoto = false
    @State private var photoFailed = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("Life stage and engorgement from one photo")
                    .font(.subheadline)
                    .foregroundStyle(Theme.muted)
                    .padding(.top, -12)

                scanCard

                if !scans.isEmpty { recent }

                Card(fill: Theme.accentTint, padding: 16) {
                    Text("Just found a tick?").font(.rounded(.headline))
                    VStack(alignment: .leading, spacing: 4) {
                        NumberedLine(1, "Pull it straight out with fine tweezers")
                        NumberedLine(2, "Keep it in a sealed bag or taped to paper")
                        NumberedLine(3, "Scan it, then watch for symptoms for 30 days")
                    }
                    Button("Read the full guide", action: openGuide)
                        .font(.subheadline.weight(.semibold))
                }

                Text("Knows 8 common U.S. ticks. Informational, not a diagnosis.")
                    .font(.caption)
                    .foregroundStyle(Theme.muted)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 24)
        }
        .background(Theme.ground)
        .navigationTitle("TickPredict")
        .overlay {
            if loadingPhoto {
                ProgressView("Opening photo")
                    .padding(20)
                    .glassEffect(.regular, in: .rect(cornerRadius: 20))
            }
        }
        .alert("Couldn't open that photo", isPresented: $photoFailed) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Try another photo, or check that it has finished downloading from iCloud.")
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            loadingPhoto = true
            Task {
                let data = try? await item.loadTransferable(type: Data.self)
                loadingPhoto = false
                pickerItem = nil
                if let data { open(PendingPhoto(data: data)) } else { photoFailed = true }
            }
        }
    }

    private var scanCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 14) {
                Image(systemName: "viewfinder")
                    .font(.system(size: 26, weight: .medium))
                    .frame(width: 52, height: 52)
                    .background(.white.opacity(0.14), in: .rect(cornerRadius: 16))
                VStack(alignment: .leading, spacing: 2) {
                    Text("Scan a tick").font(.rounded(.title2))
                    Text("Runs privately on your iPhone")
                        .font(.subheadline)
                        .opacity(0.82)
                }
            }
            HStack(spacing: 10) {
                Button { showCamera = true } label: {
                    Label("Open camera", systemImage: "camera")
                        .font(.headline)
                        .frame(maxWidth: .infinity, minHeight: 50)
                        .foregroundStyle(Color(light: 0x2E5A3A, dark: 0x2E5A3A))
                        .background(.white, in: .rect(cornerRadius: 14))
                }
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Image(systemName: "photo.on.rectangle")
                        .font(.system(size: 19, weight: .semibold))
                        .frame(width: 50, height: 50)
                        .background(.white.opacity(0.14), in: .rect(cornerRadius: 14))
                }
                .accessibilityLabel("Choose from Photos")
            }
        }
        .foregroundStyle(.white)
        .padding(22)
        .background(Color(light: 0x2E5A3A, dark: 0x24472E), in: .rect(cornerRadius: 24))
    }

    private var recent: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text("Recent").font(.rounded(.title3))
                Spacer()
                Button("See all", action: openHistory).font(.subheadline)
            }
            VStack(spacing: 0) {
                ForEach(Array(scans.prefix(2).enumerated()), id: \.element.id) { index, scan in
                    if index > 0 { Divider().overlay(Theme.line).padding(.leading, 74) }
                    NavigationLink(value: Route.saved(scan)) { ScanRow(scan: scan) }
                        .buttonStyle(.plain)
                }
            }
            .background(Theme.surface, in: .rect(cornerRadius: 18))
        }
    }
}

struct NumberedLine: View {
    let number: Int
    let text: String
    init(_ number: Int, _ text: String) { self.number = number; self.text = text }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("\(number).").monospacedDigit().foregroundStyle(Theme.accent).fontWeight(.semibold)
            Text(text)
        }
        .font(.subheadline)
        .foregroundStyle(Theme.body)
    }
}

/// One saved scan in a list: thumbnail, result, and when/where.
struct ScanRow: View {
    let scan: Scan
    var showsChevron = true

    var body: some View {
        HStack(spacing: 12) {
            ScanThumbnail(data: scan.thumbnailData, size: 50)
            VStack(alignment: .leading, spacing: 2) {
                Text(scan.title).font(.body.weight(.semibold)).foregroundStyle(Theme.ink)
                Text(subtitle).font(.footnote).foregroundStyle(Theme.muted)
                if !scan.notes.isEmpty {
                    Text(scan.notes).font(.footnote).foregroundStyle(Theme.body).lineLimit(1)
                }
            }
            Spacer(minLength: 0)
            if showsChevron {
                Image(systemName: "chevron.right")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Theme.muted.opacity(0.6))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .contentShape(.rect)
    }

    private var subtitle: String {
        let date = Calendar.current.isDateInToday(scan.foundOn)
            ? "Today" : scan.foundOn.formatted(.dateTime.month(.abbreviated).day())
        return [date, scan.bodyArea?.name, scan.place.isEmpty ? nil : scan.place]
            .compactMap { $0 }.joined(separator: " · ")
    }
}

struct ScanThumbnail: View {
    let data: Data?
    let size: CGFloat

    var body: some View {
        Group {
            if let data, let image = UIImage(data: data) {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "ladybug").foregroundStyle(Theme.muted)
            }
        }
        .frame(width: size, height: size)
        .background(Theme.photoBackdrop)
        .clipShape(.rect(cornerRadius: 12))
        .accessibilityHidden(true)
    }
}
