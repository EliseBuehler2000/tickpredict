import SwiftUI
import UIKit

/// The "calm natural" palette from the design mockup, with dark-mode pairs.
enum Theme {
    static let ground = Color(light: 0xF6F4EE, dark: 0x111512)
    static let surface = Color(light: 0xFFFFFF, dark: 0x1B201C)
    static let ink = Color(light: 0x18241C, dark: 0xEDF1EC)
    static let body = Color(light: 0x2A3A2F, dark: 0xD3DAD4)
    static let muted = Color(light: 0x56605A, dark: 0xA0A9A3)
    static let accent = Color(light: 0x2E5A3A, dark: 0x86BC93)
    static let onAccent = Color(light: 0xFFFFFF, dark: 0x0F1F14)
    static let accentTint = Color(light: 0xE3EBDD, dark: 0x1E2C22)
    static let line = Color(light: 0xECE9E1, dark: 0x2A312C)
    static let track = Color(light: 0xE9E6DD, dark: 0x2D352F)
    static let photoBackdrop = Color(light: 0xEFEBE2, dark: 0x262B27)
    static let warningFill = Color(light: 0xFBEDE6, dark: 0x3A231B)
    static let warningInk = Color(light: 0x6B2410, dark: 0xF4BFA6)
    static let bite = Color(light: 0xC2410C, dark: 0xF97B4A)

    /// Unengorged -> semi -> fully, increasing in depth.
    static let engorgementRamp: [Color] = [
        Color(light: 0xEFE4D3, dark: 0x5B4C3D),
        Color(light: 0xC98B5A, dark: 0xB57A4C),
        Color(light: 0x6E3A22, dark: 0xE7B58D),
    ]
}

// SwiftUI resolves dynamic colours on its render thread, so the provider must
// not be main-actor isolated (that traps at runtime when the screen updates).
nonisolated extension Color {
    init(light: UInt32, dark: UInt32) {
        self.init(uiColor: UIColor { @Sendable traits in
            UIColor(hex: traits.userInterfaceStyle == .dark ? dark : light)
        })
    }
}

nonisolated extension UIColor {
    convenience init(hex: UInt32) {
        self.init(red: CGFloat((hex >> 16) & 0xFF) / 255,
                  green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
    }
}

extension Font {
    static func rounded(_ style: Font.TextStyle, weight: Font.Weight = .bold) -> Font {
        .system(style, design: .rounded, weight: weight)
    }
}

/// A rounded content card on the warm background.
struct Card<Content: View>: View {
    var fill: Color = Theme.surface
    var padding: CGFloat = 18
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) { content }
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(fill, in: .rect(cornerRadius: 20))
    }
}

/// Small spaced capitals used above each value.
struct Eyebrow: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text.uppercased())
            .font(.caption.weight(.semibold))
            .tracking(0.6)
            .foregroundStyle(Theme.muted)
    }
}

struct ConfidenceBar: View {
    let value: Double
    var height: CGFloat = 6
    var tint: Color = Theme.accent

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(Theme.track)
                Capsule().fill(tint).frame(width: proxy.size.width * min(max(value, 0), 1))
            }
        }
        .frame(height: height)
        .accessibilityHidden(true)
    }
}

nonisolated extension Double {
    var percent: String { formatted(.percent.precision(.fractionLength(0))) }
}
