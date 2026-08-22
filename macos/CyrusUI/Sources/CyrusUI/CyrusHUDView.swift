import SwiftUI

struct CyrusHUDView: View {
    @ObservedObject var store: CyrusStateStore

    private var accent: Color { HUDTheme.accent(for: store.state) }

    var body: some View {
        ZStack {
            HUDBackground()
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(accent.opacity(store.state == .offline ? 0.10 : 0.26), lineWidth: 1)
                .padding(8)
                .shadow(color: accent.opacity(store.state.isActive ? 0.16 : 0), radius: 22)
                .animation(.easeInOut(duration: 0.28), value: store.state)
            VStack(spacing: 0) {
                header
                Divider().overlay(HUDTheme.cyan.opacity(0.17))
                HStack(spacing: 20) {
                    systemRail
                    centerStage
                        .frame(maxWidth: .infinity)
                    sessionRail
                }
                .padding(24)
                footer
            }
        }
        .frame(minWidth: 860, minHeight: 600)
        .preferredColorScheme(.dark)
        .animation(.easeInOut(duration: 0.24), value: store.state)
    }

    private var header: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text("C Y R U S")
                    .font(.system(size: 27, weight: .light, design: .monospaced))
                    .tracking(7)
                    .foregroundStyle(HUDTheme.primaryText)
                Text("LOCAL INTELLIGENCE AND OPERATIONS SYSTEM")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(HUDTheme.secondaryText)
            }
            Spacer()
            if store.isDemo {
                statusBadge(label: "DEMO FEED", symbol: "eye", color: HUDTheme.amber)
            }
            statusBadge(
                label: store.state.title,
                symbol: store.state.symbolName,
                color: accent
            )
        }
        .padding(.horizontal, 26)
        .padding(.vertical, 18)
    }

    private var systemRail: some View {
        VStack(alignment: .leading, spacing: 16) {
            railTitle("SYSTEMS", symbol: "square.stack.3d.up")
            metric("VOICE RUNTIME", value: store.state == .offline ? "OFFLINE" : "LIVE", ready: store.state != .offline)
            metric("STATE FEED", value: store.state == .offline ? "STALE" : "OWNER-ONLY", ready: store.state != .offline)
            metric("UI EGRESS", value: "NONE", ready: true)
            metric("UI PAYLOAD", value: "CONTENT-FREE", ready: true)
            Spacer(minLength: 8)
            HStack(spacing: 7) {
                Image(systemName: "lock.shield")
                Text("LOCAL-FIRST BOUNDARY")
            }
            .font(.system(size: 10, weight: .semibold, design: .monospaced))
            .foregroundStyle(HUDTheme.mint)
        }
        .frame(width: 210)
        .frame(maxHeight: 390, alignment: .top)
        .hudPanel(accent: store.state == .offline ? HUDTheme.danger : HUDTheme.cyan)
    }

    private var centerStage: some View {
        VStack(spacing: 15) {
            ArcReactorView(state: store.state, activity: store.activity)
                .frame(width: 278, height: 278)
            VStack(spacing: 7) {
                Text(store.state.subtitle)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(HUDTheme.primaryText)
                    .multilineTextAlignment(.center)
                    .id(store.state)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                VoiceWaveformView(state: store.state, activity: store.activity)
                ProcessingPipelineView(state: store.state, stateAge: store.stateAge)
                Text("GEN \(String(format: "%06d", store.generation))  •  \(formattedStateAge)")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .tracking(1.2)
                    .foregroundStyle(HUDTheme.secondaryText.opacity(0.76))
                    .contentTransition(.numericText())
            }
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .contain)
    }

    private var sessionRail: some View {
        VStack(alignment: .leading, spacing: 16) {
            railTitle("SESSION", symbol: "dot.radiowaves.left.and.right")
            infoBlock(label: "ACTIVATION", value: "CONTROL + OPTION", symbol: "keyboard")
            infoBlock(label: "MODEL DETAILS", value: "NOT EXPOSED", symbol: "eye.slash")
            infoBlock(label: "VOICE MODE", value: "PUSH TO TALK", symbol: "mic")
            infoBlock(label: "DISPLAY DATA", value: "NO TRANSCRIPTS", symbol: "text.badge.xmark")
            LiveLevelMeter(activity: store.state == .listening ? store.activity : 0)
            Spacer(minLength: 8)
            Text("The HUD observes operational state only. Requests and responses remain inside the Voice and Orchestrator boundary.")
                .font(.system(size: 10, weight: .regular))
                .foregroundStyle(HUDTheme.secondaryText)
                .lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(width: 210)
        .frame(maxHeight: 390, alignment: .top)
        .hudPanel(accent: accent)
    }

    private var footer: some View {
        HStack {
            Label("MACOS NATIVE HUD", systemImage: "macwindow")
            Spacer()
            Text("MARK 0  •  LOCAL OPERATOR INTERFACE")
        }
        .font(.system(size: 9, weight: .semibold, design: .monospaced))
        .tracking(1.1)
        .foregroundStyle(HUDTheme.secondaryText.opacity(0.82))
        .padding(.horizontal, 26)
        .padding(.vertical, 13)
        .background(Color.black.opacity(0.2))
    }

    private func statusBadge(label: String, symbol: String, color: Color) -> some View {
        Label(label, systemImage: symbol)
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .tracking(1.1)
            .foregroundStyle(color)
            .padding(.horizontal, 12)
            .frame(height: 32)
            .background(Capsule().fill(color.opacity(0.11)))
            .overlay(Capsule().stroke(color.opacity(0.42), lineWidth: 1))
            .accessibilityLabel("Status: \(label)")
    }

    private func railTitle(_ text: String, symbol: String) -> some View {
        Label(text, systemImage: symbol)
            .font(.system(size: 11, weight: .bold, design: .monospaced))
            .tracking(1.5)
            .foregroundStyle(HUDTheme.cyan)
    }

    private func metric(_ label: String, value: String, ready: Bool) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(HUDTheme.secondaryText)
            HStack(spacing: 7) {
                Circle().fill(ready ? HUDTheme.mint : HUDTheme.danger).frame(width: 6, height: 6)
                Text(value)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(HUDTheme.primaryText)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func infoBlock(label: String, value: String, symbol: String) -> some View {
        HStack(spacing: 11) {
            Image(systemName: symbol)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(HUDTheme.cyan)
                .frame(width: 30, height: 30)
                .background(RoundedRectangle(cornerRadius: 8).fill(HUDTheme.cyan.opacity(0.09)))
            VStack(alignment: .leading, spacing: 3) {
                Text(label)
                    .font(.system(size: 8, weight: .medium, design: .monospaced))
                    .foregroundStyle(HUDTheme.secondaryText)
                Text(value)
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(HUDTheme.primaryText)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var formattedStateAge: String {
        let total = max(0, Int(store.stateAge))
        return String(format: "%02d:%02d IN STATE", total / 60, total % 60)
    }
}

private struct HUDBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [HUDTheme.canvas, Color(red: 0.006, green: 0.018, blue: 0.03)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            Canvas { context, size in
                let spacing: CGFloat = 32
                var path = Path()
                stride(from: 0, through: size.width, by: spacing).forEach { x in
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: size.height))
                }
                stride(from: 0, through: size.height, by: spacing).forEach { y in
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: size.width, y: y))
                }
                context.stroke(path, with: .color(HUDTheme.grid), lineWidth: 0.5)
            }
            RadialGradient(
                colors: [HUDTheme.blue.opacity(0.10), .clear],
                center: .center,
                startRadius: 80,
                endRadius: 440
            )
        }
        .ignoresSafeArea()
    }
}
