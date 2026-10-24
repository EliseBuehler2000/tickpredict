@preconcurrency import AVFoundation
import Observation
import SwiftUI

/// Owns the capture session. All AVFoundation work happens on one serial queue.
nonisolated final class CameraSession: NSObject, @unchecked Sendable {
    let session = AVCaptureSession()
    private let output = AVCapturePhotoOutput()
    private let queue = DispatchQueue(label: "TickPredict.camera")
    private var configured = false
    private var pending: [Int64: CheckedContinuation<Data, Error>] = [:]

    enum Failure: Error { case unavailable, captureFailed }

    /// Prefers multi-camera devices so iPhones with macro switch to it automatically up close.
    private static func bestDevice() -> AVCaptureDevice? {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInWideAngleCamera],
            mediaType: .video, position: .back
        ).devices.first
    }

    func start() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async { [self] in
                do {
                    if !configured { try configure() }
                    if !session.isRunning { session.startRunning() }
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func stop() {
        queue.async { [self] in
            if session.isRunning { session.stopRunning() }
        }
    }

    private func configure() throws {
        guard let device = Self.bestDevice(),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input), session.canAddOutput(output)
        else { throw Failure.unavailable }
        session.beginConfiguration()
        session.sessionPreset = .photo
        session.addInput(input)
        session.addOutput(output)
        output.maxPhotoQualityPrioritization = .quality
        session.commitConfiguration()
        if (try? device.lockForConfiguration()) != nil {
            if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
            device.unlockForConfiguration()
        }
        configured = true
    }

    func capture(flash: Bool) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                // Capturing without a live connection (e.g. after a phone call
                // interrupted the session) raises an exception, so check first.
                guard session.isRunning, let connection = output.connection(with: .video),
                      connection.isActive, connection.isEnabled
                else {
                    continuation.resume(throwing: Failure.unavailable)
                    return
                }
                let settings = AVCapturePhotoSettings()
                settings.photoQualityPrioritization = .quality
                if output.supportedFlashModes.contains(.on) { settings.flashMode = flash ? .on : .off }
                pending[settings.uniqueID] = continuation
                output.capturePhoto(with: settings, delegate: self)
            }
        }
    }
}

extension CameraSession: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let data = photo.fileDataRepresentation()
        let id = photo.resolvedSettings.uniqueID
        queue.async { [self] in
            guard let continuation = pending.removeValue(forKey: id) else { return }
            if let data, error == nil {
                continuation.resume(returning: data)
            } else {
                continuation.resume(throwing: error ?? Failure.captureFailed)
            }
        }
    }
}

@Observable
final class CameraModel {
    enum Status { case starting, running, denied, unavailable }

    private(set) var status = Status.starting
    private(set) var isCapturing = false
    private(set) var captures = 0
    var flashOn = false
    let camera = CameraSession()

    func start() async {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: break
        case .notDetermined:
            guard await AVCaptureDevice.requestAccess(for: .video) else { status = .denied; return }
        default:
            status = .denied
            return
        }
        do {
            try await camera.start()
            status = .running
        } catch {
            status = .unavailable
        }
    }

    func stop() { camera.stop() }

    /// Returns nil if a capture is already in progress; throws if the camera failed.
    func capture() async throws -> Data? {
        guard status == .running, !isCapturing else { return nil }
        isCapturing = true
        captures += 1
        defer { isCapturing = false }
        return try await camera.capture(flash: flashOn)
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}
}
