import SwiftUI

struct ArcReactorView: View {
    let state: CyrusRuntimeState
    let activity: Double
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: !motionActive)) { timeline in
            let phase = motionActive ? timeline.date.timeIntervalSinceReferenceDate : 0.0
            ZStack {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [accent.opacity(0.42), accent.opacity(0.08), .clear],
                            center: .center,
                            startRadius: 5,
                            endRadius: 145
                        )
                    )
                    .blur(radius: 16)
                    .scaleEffect(coreScale(phase: phase))

                reactorRing(segments: 12, width: 3, trim: 0.055)
                    .rotationEffect(.degrees(phase * 18))
                    .frame(width: 250, height: 250)
                reactorRing(segments: 8, width: 2, trim: 0.075)
                    .rotationEffect(.degrees(-phase * 27))
                    .frame(width: 205, height: 205)
                Circle()
                    .stroke(accent.opacity(0.34), lineWidth: 1)
                    .frame(width: 166, height: 166)
                Circle()
                    .stroke(accent.opacity(0.9), lineWidth: 2)
                    .frame(width: 132, height: 132)
                    .shadow(color: accent.opacity(0.9), radius: 12)

                if state == .listening {
                    Circle()
                        .trim(from: 0, to: max(0.025, activity))
                        .stroke(
                            accent,
                            style: StrokeStyle(lineWidth: 5, lineCap: .round)
                        )
                        .rotationEffect(.degrees(-90))
                        .frame(width: 148, height: 148)
                        .shadow(color: accent, radius: 8 + CGFloat(activity) * 12)
                        .animation(.spring(response: 0.18, dampingFraction: 0.72), value: activity)
                }

                stateDecoration(phase: phase)

                VStack(spacing: 8) {
                    Image(systemName: state.symbolName)
                        .font(.system(size: 34, weight: .light))
                        .symbolEffect(.pulse, options: .repeating, isActive: motionActive)
                    Text(state.title)
                        .font(.system(size: 13, weight: .semibold, design: .monospaced))
                        .tracking(2.2)
                }
                .foregroundStyle(accent)
            }
            .animation(.easeInOut(duration: 0.28), value: state)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("CYRUS status: \(state.title)")
        .accessibilityValue(state.subtitle)
    }

    private var accent: Color { HUDTheme.accent(for: state) }

    private func coreScale(phase: TimeInterval) -> CGFloat {
        if state == .listening { return 0.98 + activity * 0.09 }
        if state == .speaking && motionActive {
            return 1.01 + CGFloat(abs(sin(phase * 3.2))) * 0.035
        }
        return state == .offline ? 0.92 : 0.98
    }

    private var motionActive: Bool {
        guard !reduceMotion else { return false }
        return [.starting, .listening, .processing, .speaking].contains(state)
    }

    private func reactorRing(segments: Int, width: CGFloat, trim: CGFloat) -> some View {
        ZStack {
            ForEach(0..<segments, id: \.self) { index in
                Circle()
                    .trim(from: 0, to: trim)
                    .stroke(
                        accent.opacity(index.isMultiple(of: 3) ? 0.95 : 0.46),
                        style: StrokeStyle(lineWidth: width, lineCap: .round)
                    )
                    .rotationEffect(.degrees(Double(index) * (360.0 / Double(segments))))
            }
        }
    }

    @ViewBuilder
    private func stateDecoration(phase: TimeInterval) -> some View {
        switch state {
        case .processing, .starting:
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(accent)
                    .frame(width: 7, height: 7)
                    .offset(y: -94)
                    .rotationEffect(.degrees(phase * 64 + Double(index) * 120))
                    .shadow(color: accent, radius: 6)
            }
        case .speaking:
            ForEach(0..<3, id: \.self) { index in
                let progress = motionActive
                    ? (phase * 0.72 + Double(index) / 3).truncatingRemainder(dividingBy: 1)
                    : Double(index + 1) / 4
                Circle()
                    .stroke(accent.opacity(1 - progress), lineWidth: 1.5)
                    .frame(width: 150, height: 150)
                    .scaleEffect(1 + CGFloat(progress) * 0.62)
            }
        case .error:
            Image(systemName: "bolt.trianglebadge.exclamationmark.fill")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(accent)
                .offset(y: -102)
        default:
            EmptyView()
        }
    }
}

struct VoiceWaveformView: View {
    let state: CyrusRuntimeState
    let activity: Double
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 24.0, paused: reduceMotion || state != .speaking)) { timeline in
            let time = reduceMotion ? 0 : timeline.date.timeIntervalSinceReferenceDate
            HStack(alignment: .center, spacing: 5) {
                ForEach(0..<13, id: \.self) { index in
                    Capsule()
                        .fill(HUDTheme.accent(for: state).opacity(0.38 + Double(index % 3) * 0.2))
                        .frame(width: 3, height: barHeight(index: index, time: time))
                        .animation(.spring(response: 0.18, dampingFraction: 0.68), value: activity)
                }
            }
            .frame(height: 36)
        }
        .accessibilityHidden(true)
    }

    private func barHeight(index: Int, time: TimeInterval) -> CGFloat {
        if state == .listening {
            let shape = 0.46 + abs(sin(Double(index) * 1.21)) * 0.54
            return 5 + CGFloat(activity * shape) * 30
        }
        if state == .speaking {
            let wave = abs(sin(time * 4.6 + Double(index) * 0.72))
            return 7 + CGFloat(wave) * 27
        }
        return index.isMultiple(of: 2) ? 5 : 9
    }
}
