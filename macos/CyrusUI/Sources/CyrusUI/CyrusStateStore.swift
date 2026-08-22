import Combine
import Foundation

@MainActor
final class CyrusStateStore: NSObject, ObservableObject {
    @Published private(set) var state: CyrusRuntimeState = .offline
    @Published private(set) var generation = 0
    @Published private(set) var activity = 0.0
    @Published private(set) var stateAge = 0.0
    @Published private(set) var isDemo = false

    private let stateURL: URL
    private let now: () -> Date
    private var timer: Timer?
    private var demoIndex = 0

    init(
        stateURL: URL = CyrusUIArguments.stateFile(),
        demo: Bool = CyrusUIArguments.demoEnabled(),
        now: @escaping () -> Date = Date.init,
        startTimer: Bool = true
    ) {
        self.stateURL = stateURL
        self.isDemo = demo
        self.now = now
        super.init()
        refresh()
        if startTimer {
            timer = Timer.scheduledTimer(
                timeInterval: 0.1,
                target: self,
                selector: #selector(timerFired),
                userInfo: nil,
                repeats: true
            )
            if let timer { RunLoop.main.add(timer, forMode: .common) }
        }
    }

    @objc private func timerFired() { refresh() }

    func refresh() {
        if isDemo {
            let states: [CyrusRuntimeState] = [.idle, .listening, .processing, .speaking]
            let timestamp = now().timeIntervalSince1970
            let next = Int(timestamp / 2.5) % states.count
            if next != demoIndex || state == .offline {
                demoIndex = next
                state = states[next]
                generation += 1
                stateAge = 0
            } else {
                stateAge += 0.1
            }
            activity = state == .listening
                ? 0.18 + abs(sin(timestamp * 4.2)) * 0.78
                : state == .speaking
                ? 0.25 + abs(sin(timestamp * 3.1)) * 0.62
                : 0
            return
        }
        guard let data = PrivateSnapshotReader.read(from: stateURL),
              let envelope = VoiceStateEnvelope.decodeStrict(from: data),
              let liveState = envelope.validated(
                nowUnixMs: Int64(now().timeIntervalSince1970 * 1_000)
              )
        else {
            state = .offline
            activity = 0
            stateAge = 0
            return
        }
        state = liveState
        generation = envelope.generation
        activity = envelope.activity
        stateAge = max(
            0,
            now().timeIntervalSince1970
                - Double(envelope.stateStartedAtUnixMs) / 1_000.0
        )
    }
}
