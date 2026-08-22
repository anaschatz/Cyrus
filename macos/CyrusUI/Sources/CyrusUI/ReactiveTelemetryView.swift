import SwiftUI

struct ProcessingPipelineView: View {
    let state: CyrusRuntimeState
    let stateAge: TimeInterval
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let stages = [
        ("HEAR", "waveform"),
        ("REASON", "cpu"),
        ("VOICE", "speaker.wave.2"),
    ]

    var body: some View {
        HStack(spacing: 7) {
            ForEach(Array(stages.enumerated()), id: \.offset) { index, stage in
                stageView(index: index, label: stage.0, symbol: stage.1)
                if index < stages.count - 1 {
                    Capsule()
                        .fill(HUDTheme.cyan.opacity(stageIndex > index ? 0.72 : 0.14))
                        .frame(width: 18, height: 1)
                }
            }
        }
        .animation(.easeOut(duration: 0.24), value: stageIndex)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Processing stage \(stageLabel)")
    }

    private var stageIndex: Int {
        switch state {
        case .listening: 0
        case .processing: 1
        case .speaking: 2
        default: -1
        }
    }

    private var stageLabel: String {
        guard stages.indices.contains(stageIndex) else { return "inactive" }
        return stages[stageIndex].0.lowercased()
    }

    private func stageView(index: Int, label: String, symbol: String) -> some View {
        let active = index == stageIndex
        let complete = index < stageIndex
        let pulse = active && state == .processing && !reduceMotion
            ? 1 + CGFloat(abs(sin(stateAge * 3.2))) * 0.05
            : 1
        return VStack(spacing: 5) {
            Image(systemName: complete ? "checkmark" : symbol)
                .font(.system(size: 10, weight: .semibold))
                .frame(width: 28, height: 28)
                .background(
                    Circle().fill(
                        active
                            ? HUDTheme.cyan.opacity(0.18)
                            : HUDTheme.surface.opacity(0.72)
                    )
                )
                .overlay(
                    Circle().stroke(
                        complete ? HUDTheme.mint : active ? HUDTheme.cyan : HUDTheme.cyan.opacity(0.14),
                        lineWidth: active ? 1.5 : 1
                    )
                )
            Text(label)
                .font(.system(size: 7, weight: .bold, design: .monospaced))
                .tracking(0.8)
        }
        .foregroundStyle(complete ? HUDTheme.mint : active ? HUDTheme.cyan : HUDTheme.secondaryText.opacity(0.5))
        .scaleEffect((active ? 1.06 : 1) * pulse)
    }
}

struct LiveLevelMeter: View {
    let activity: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("INPUT ACTIVITY")
                Spacer()
                Text("\(Int(activity * 100))%")
                    .contentTransition(.numericText())
            }
            .font(.system(size: 8, weight: .semibold, design: .monospaced))
            .foregroundStyle(HUDTheme.secondaryText)
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(HUDTheme.cyan.opacity(0.10))
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [HUDTheme.blue, HUDTheme.cyan, HUDTheme.mint],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: max(3, proxy.size.width * CGFloat(activity)))
                        .shadow(color: HUDTheme.cyan.opacity(activity), radius: 5)
                }
            }
            .frame(height: 4)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Input activity \(Int(activity * 100)) percent")
    }
}
