import CoreLocation
import MapKit
import SwiftUI

/// Finds nearby care with Apple Maps. The location is used for the search only and never stored.
@Observable
final class CareFinder {
    enum Kind: String, CaseIterable, Identifiable {
        case urgent = "Urgent care", doctors = "Doctors", hospitals = "Hospitals"
        var id: String { rawValue }

        var query: String {
            switch self {
            case .urgent: "urgent care"
            case .doctors: "primary care doctor"
            case .hospitals: "hospital"
            }
        }
    }

    enum Status: Equatable {
        case locating, searching, ready
        /// Location permission is off or restricted.
        case denied
        /// No location fix arrived in time.
        case noLocation
        /// A typed town or ZIP code wasn't recognised.
        case placeNotFound
        /// Apple Maps couldn't be reached.
        case offline
    }

    struct Place: Identifiable {
        let id = UUID()
        let item: MKMapItem
        let coordinate: CLLocationCoordinate2D
        let distance: CLLocationDistance
        var name: String { item.name ?? "Unnamed place" }
        var address: String? { item.address?.shortAddress }
        var phoneURL: URL? {
            guard let phone = item.phoneNumber else { return nil }
            let digits = phone.filter { $0.isNumber || $0 == "+" }
            return digits.isEmpty ? nil : URL(string: "tel:\(digits)")
        }
    }

    var kind = Kind.urgent
    private(set) var status = Status.locating
    private(set) var origin: CLLocation?
    private(set) var places: [Place] = []
    private var searchID = 0

    func locate() async {
        status = .locating
        let session = CLServiceSession(authorization: .whenInUse)
        let found = await withTaskGroup(of: CLLocation?.self) { group in
            group.addTask {
                do {
                    for try await update in CLLocationUpdate.liveUpdates() {
                        if update.authorizationDenied || update.authorizationDeniedGlobally { return nil }
                        if let location = update.location { return location }
                    }
                } catch {}
                return nil
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(20))
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
        withExtendedLifetime(session) {}
        guard let found else {
            let authorization = CLLocationManager().authorizationStatus
            status = authorization == .denied || authorization == .restricted ? .denied : .noLocation
            return
        }
        origin = found
        await search()
    }

    /// Searches around a typed town or ZIP code instead of the current location.
    func useTyped(_ text: String) async {
        status = .searching
        let request = MKLocalSearch.Request()
        request.naturalLanguageQuery = text
        request.resultTypes = .address
        do {
            guard let item = try await MKLocalSearch(request: request).start().mapItems.first else {
                status = .placeNotFound
                return
            }
            origin = item.location
            await search()
        } catch {
            status = Self.isNotFound(error) ? .placeNotFound : .offline
        }
    }

    private static func isNotFound(_ error: Error) -> Bool {
        (error as? MKError)?.code == .placemarkNotFound
    }

    /// Apple Maps can return vets for "urgent care"; TickPredict is for people.
    private static func isForAnimals(_ item: MKMapItem) -> Bool {
        if item.pointOfInterestCategory == .animalService { return true }
        let name = (item.name ?? "").lowercased()
        return ["pet", "vet", "animal"].contains { name.split(whereSeparator: { !$0.isLetter }).contains(Substring($0)) }
            || name.contains("veterinar")
    }

    func search() async {
        guard let origin else { return }
        // Switching the care type quickly overlaps searches; only the latest may update.
        searchID += 1
        let id = searchID
        status = .searching
        let request = MKLocalSearch.Request()
        request.naturalLanguageQuery = kind.query
        request.resultTypes = .pointOfInterest
        request.region = MKCoordinateRegion(center: origin.coordinate, latitudinalMeters: 25_000, longitudinalMeters: 25_000)
        request.pointOfInterestFilter = kind == .hospitals
            ? MKPointOfInterestFilter(including: [.hospital])
            : MKPointOfInterestFilter(excluding: [.animalService])
        do {
            let items = try await MKLocalSearch(request: request).start().mapItems
            guard id == searchID else { return }
            places = items.filter { !Self.isForAnimals($0) }.map { item in
                Place(item: item, coordinate: item.location.coordinate, distance: item.location.distance(from: origin))
            }
            .sorted { $0.distance < $1.distance }
            .prefix(8)
            .map { $0 }
            status = .ready
        } catch {
            guard id == searchID else { return }
            places = []
            status = Self.isNotFound(error) ? .ready : .offline
        }
    }
}

struct FindCareView: View {
    @State private var finder = CareFinder()
    @State private var camera = MapCameraPosition.automatic
    @State private var typed = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                map

                Picker("Type of care", selection: $finder.kind) {
                    ForEach(CareFinder.Kind.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .onChange(of: finder.kind) { Task { await finder.search() } }

                content

                Link(destination: URL(string: "tel:911")!) {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "exclamationmark.triangle")
                        Text("Trouble breathing, a stiff neck, or facial drooping? **Call 911.**")
                            .multilineTextAlignment(.leading)
                        Spacer(minLength: 0)
                        Image(systemName: "phone.fill")
                    }
                    .font(.subheadline)
                    .foregroundStyle(Theme.warningInk)
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Theme.warningFill, in: .rect(cornerRadius: 16))
                }
                .accessibilityLabel("Emergency: trouble breathing, a stiff neck, or facial drooping. Call 911.")

                Text("Places from Apple Maps. Your location is used only for this search and is never stored.")
                    .font(.caption)
                    .foregroundStyle(Theme.muted)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
        }
        .background(Theme.ground)
        .navigationTitle("Find care")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            // Runs again when returning to this screen; keep earlier results.
            if finder.origin == nil { await finder.locate() }
        }
    }

    private var map: some View {
        Map(position: $camera) {
            UserAnnotation()
            ForEach(Array(finder.places.enumerated()), id: \.element.id) { index, place in
                Marker(place.name, monogram: Text("\(index + 1)"), coordinate: place.coordinate)
                    .tint(Theme.accent)
            }
        }
        .mapStyle(.standard(pointsOfInterest: .excludingAll))
        .frame(height: 240)
        .clipShape(.rect(cornerRadius: 20))
        .overlay(alignment: .topTrailing) {
            Button {
                Task { await finder.locate() }
            } label: {
                Image(systemName: "location.fill").frame(width: 40, height: 40)
            }
            .glassEffect(.regular.interactive(), in: .circle)
            .padding(10)
            .accessibilityLabel("Use my location")
        }
        .onChange(of: finder.places.map(\.id)) { withAnimation { camera = .automatic } }
    }

    @ViewBuilder private var content: some View {
        switch finder.status {
        case .locating, .searching:
            HStack(spacing: 10) {
                ProgressView()
                Text(finder.status == .locating ? "Finding your location" : "Searching nearby")
                    .foregroundStyle(Theme.muted)
            }
            .frame(maxWidth: .infinity, minHeight: 80)
        case .offline:
            Card {
                Text("Couldn't reach Apple Maps").font(.headline)
                Text("Check your internet connection and try again.")
                    .font(.subheadline).foregroundStyle(Theme.body)
                Button("Try again") { Task { await retry() } }.buttonStyle(.glassProminent)
            }
        case .denied, .noLocation, .placeNotFound:
            Card {
                Text(finder.status == .placeNotFound ? "Couldn't find that place" : "Search by town instead")
                    .font(.headline)
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(Theme.body)
                HStack {
                    TextField("Town or ZIP code", text: $typed)
                        .submitLabel(.search)
                        .onSubmit(searchTyped)
                        .padding(.horizontal, 12)
                        .frame(minHeight: 44)
                        .background(Theme.ground, in: .rect(cornerRadius: 12))
                    Button("Search", action: searchTyped).buttonStyle(.glassProminent)
                }
                if finder.status == .denied, let settings = URL(string: UIApplication.openSettingsURLString) {
                    Link("Turn on location in Settings", destination: settings).font(.subheadline.weight(.semibold))
                } else if finder.status == .noLocation {
                    Button("Try my location again") { Task { await finder.locate() } }
                        .font(.subheadline.weight(.semibold))
                }
            }
        case .ready where finder.places.isEmpty:
            Card {
                Text("Nothing found nearby").font(.headline)
                Text("Try another type of care, or call your doctor's office.")
                    .font(.subheadline).foregroundStyle(Theme.body)
            }
        case .ready:
            VStack(spacing: 0) {
                ForEach(Array(finder.places.enumerated()), id: \.element.id) { index, place in
                    if index > 0 { Divider().overlay(Theme.line).padding(.leading, 54) }
                    PlaceRow(number: index + 1, place: place)
                }
            }
            .background(Theme.surface, in: .rect(cornerRadius: 18))
        }
    }

    private var message: String {
        switch finder.status {
        case .denied: "Location access is off. Type a town or ZIP code to search there."
        case .placeNotFound: "Check the spelling, or try a ZIP code."
        default: "Your location isn't available right now. Type a town or ZIP code to search there."
        }
    }

    private func retry() async {
        if finder.origin == nil { await finder.locate() } else { await finder.search() }
    }

    private func searchTyped() {
        let text = typed.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }
        Task { await finder.useTyped(text) }
    }
}

private struct PlaceRow: View {
    let number: Int
    let place: CareFinder.Place

    var body: some View {
        HStack(spacing: 12) {
            Text("\(number)")
                .font(.footnote.weight(.bold))
                .foregroundStyle(Theme.accent)
                .frame(width: 28, height: 28)
                .background(Theme.accentTint, in: .circle)
            VStack(alignment: .leading, spacing: 2) {
                Text(place.name).font(.body.weight(.semibold)).foregroundStyle(Theme.ink).lineLimit(2)
                Text([Measurement(value: place.distance, unit: UnitLength.meters)
                        .formatted(.measurement(width: .abbreviated, usage: .road)), place.address]
                        .compactMap { $0 }.joined(separator: " · "))
                    .font(.footnote)
                    .foregroundStyle(Theme.muted)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            if let phone = place.phoneURL {
                Link(destination: phone) {
                    Image(systemName: "phone.fill").frame(width: 40, height: 40)
                }
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
                .accessibilityLabel("Call \(place.name)")
            }
            Button {
                place.item.openInMaps(launchOptions: [MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDefault])
            } label: {
                Image(systemName: "arrow.triangle.turn.up.right.diamond.fill").frame(width: 40, height: 40)
            }
            .buttonStyle(.glassProminent)
            .buttonBorderShape(.circle)
            .accessibilityLabel("Directions to \(place.name)")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }
}
