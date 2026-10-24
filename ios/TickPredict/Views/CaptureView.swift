import PhotosUI
import SwiftUI

struct CaptureView: View {
    let onCapture: (Data) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var model = CameraModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var flash = false
    @State private var loadingPhoto = false
    @State private var failure: String?

    private let reticle: CGFloat = 250

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            viewfinder.ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                Spacer()
                bottomBar
            }
            .padding(.horizontal, 20)

            if flash { Color.white.ignoresSafeArea().transition(.opacity) }
        }
        .foregroundStyle(.white)
        .statusBarHidden()
        .overlay {
            if loadingPhoto {
                ProgressView("Opening photo").padding(20).glassEffect(.regular, in: .rect(cornerRadius: 20))
            }
        }
        .alert(failure ?? "", isPresented: Binding(get: { failure != nil }, set: { if !$0 { failure = nil } })) {
            Button("OK", role: .cancel) {}
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: model.captures)
        .task {
            await model.start()
            // Closed before the camera finished starting.
            if Task.isCancelled { model.stop() }
        }
        .onDisappear { model.stop() }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            loadingPhoto = true
            Task {
                let data = try? await item.loadTransferable(type: Data.self)
                loadingPhoto = false
                pickerItem = nil
                if let data { finish(data) } else { failure = "Couldn't open that photo" }
            }
        }
    }

    @ViewBuilder private var viewfinder: some View {
        switch model.status {
        case .running, .starting:
            GeometryReader { proxy in
                ZStack {
                    CameraPreview(session: model.camera.session)
                    // Dim everything outside the framing circle.
                    Rectangle()
                        .fill(.black.opacity(0.38))
                        .reverseMask {
                            Circle().frame(width: reticle, height: reticle)
                                .position(x: proxy.size.width / 2, y: proxy.size.height * 0.42)
                        }
                    Circle()
                        .strokeBorder(.white.opacity(0.9), lineWidth: 2)
                        .frame(width: reticle, height: reticle)
                        .position(x: proxy.size.width / 2, y: proxy.size.height * 0.42)
                    Text("Center the tick in the circle")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .glassEffect(.regular, in: .capsule)
                        .position(x: proxy.size.width / 2, y: proxy.size.height * 0.42 + reticle / 2 + 34)
                }
            }
        case .denied:
            unavailable("Camera access is off",
                        "Allow camera access in Settings, or choose a photo you already took.",
                        settings: true)
        case .unavailable:
            unavailable("Camera isn't available",
                        "Choose a photo of the tick from your library instead.",
                        settings: false)
        }
    }

    private func unavailable(_ title: String, _ message: String, settings: Bool) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "camera.badge.ellipsis").font(.system(size: 40))
            Text(title).font(.rounded(.title3))
            Text(message).font(.subheadline).multilineTextAlignment(.center).opacity(0.8)
            if settings, let url = URL(string: UIApplication.openSettingsURLString) {
                Link("Open Settings", destination: url).buttonStyle(.glass).padding(.top, 4)
            }
        }
        .padding(40)
    }

    private var topBar: some View {
        HStack {
            Button { dismiss() } label: {
                Image(systemName: "xmark").font(.system(size: 17, weight: .semibold)).frame(width: 44, height: 44)
            }
            .glassEffect(.regular.interactive(), in: .circle)
            .accessibilityLabel("Close camera")
            Spacer()
            if model.status == .running {
                Button { model.flashOn.toggle() } label: {
                    Image(systemName: model.flashOn ? "bolt.fill" : "bolt.slash.fill")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(model.flashOn ? .yellow : .white)
                        .frame(width: 44, height: 44)
                }
                .glassEffect(.regular.interactive(), in: .circle)
                .accessibilityLabel("Flash")
                .accessibilityValue(model.flashOn ? "On" : "Off")
            }
        }
        .padding(.top, 8)
    }

    private var bottomBar: some View {
        VStack(spacing: 22) {
            GlassEffectContainer(spacing: 8) {
                FlowLayout(spacing: 8, centered: true) {
                    ForEach(["White paper", "Bright light", "Get close"], id: \.self) { tip in
                        Text(tip)
                            .font(.footnote.weight(.medium))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .glassEffect(.regular, in: .capsule)
                    }
                }
            }
            HStack {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Image(systemName: "photo.on.rectangle")
                        .font(.system(size: 20, weight: .semibold))
                        .frame(width: 52, height: 52)
                }
                .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 14))
                .accessibilityLabel("Choose from Photos")
                .frame(maxWidth: .infinity, alignment: .leading)

                Button(action: shoot) {
                    Circle()
                        .fill(.white)
                        .frame(width: 62, height: 62)
                        .padding(5)
                        .overlay(Circle().strokeBorder(.white, lineWidth: 4))
                        .scaleEffect(model.isCapturing ? 0.9 : 1)
                }
                .buttonStyle(.plain)
                .disabled(model.status != .running)
                .opacity(model.status == .running ? 1 : 0.4)
                .accessibilityLabel("Take photo")

                Text("8 common U.S. ticks")
                    .font(.caption)
                    .multilineTextAlignment(.trailing)
                    .opacity(0.75)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .padding(.bottom, 12)
    }

    private func shoot() {
        Task {
            withAnimation(.easeOut(duration: 0.08)) { flash = true }
            do {
                let data = try await model.capture()
                withAnimation(.easeIn(duration: 0.2)) { flash = false }
                if let data { finish(data) }
            } catch {
                withAnimation(.easeIn(duration: 0.2)) { flash = false }
                failure = "Couldn't take the photo. Try again."
            }
        }
    }

    private func finish(_ data: Data) {
        dismiss()
        onCapture(data)
    }
}

extension View {
    /// Cuts the given shape out of this view.
    func reverseMask<Mask: View>(@ViewBuilder _ mask: () -> Mask) -> some View {
        self.mask {
            Rectangle().overlay { mask().blendMode(.destinationOut) }.compositingGroup()
        }
    }
}
