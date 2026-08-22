import SwiftUI

enum HUDTheme {
    static let canvas = Color(red: 0.012, green: 0.035, blue: 0.052)
    static let surface = Color(red: 0.025, green: 0.075, blue: 0.105)
    static let cyan = Color(red: 0.25, green: 0.94, blue: 1.0)
    static let blue = Color(red: 0.17, green: 0.55, blue: 1.0)
    static let mint = Color(red: 0.35, green: 1.0, blue: 0.78)
    static let amber = Color(red: 1.0, green: 0.72, blue: 0.28)
    static let danger = Color(red: 1.0, green: 0.35, blue: 0.31)
    static let primaryText = Color.white.opacity(0.96)
    static let secondaryText = Color(red: 0.66, green: 0.82, blue: 0.88)
    static let grid = cyan.opacity(0.075)

    static func accent(for state: CyrusRuntimeState) -> Color {
        switch state {
        case .error: danger
        case .offline: Color.white.opacity(0.42)
        case .starting, .processing: blue
        case .idle: mint
        case .listening, .speaking: cyan
        }
    }
}

struct HUDPanelModifier: ViewModifier {
    let accent: Color

    func body(content: Content) -> some View {
        content
            .padding(18)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(HUDTheme.surface.opacity(0.54))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(accent.opacity(0.22), lineWidth: 1)
                    )
            )
    }
}

extension View {
    func hudPanel(accent: Color = HUDTheme.cyan) -> some View {
        modifier(HUDPanelModifier(accent: accent))
    }
}
