import Darwin
import Foundation

enum CyrusRuntimeState: String, Codable, CaseIterable, Sendable {
    case starting
    case idle
    case listening
    case processing
    case speaking
    case error
    case offline

    var title: String {
        switch self {
        case .starting: "INITIALIZING"
        case .idle: "ONLINE"
        case .listening: "LISTENING"
        case .processing: "PROCESSING"
        case .speaking: "RESPONDING"
        case .error: "ATTENTION"
        case .offline: "OFFLINE"
        }
    }

    var subtitle: String {
        switch self {
        case .starting: "Local systems are coming online"
        case .idle: "Hold Control + Option to speak"
        case .listening: "Listening while PTT is held"
        case .processing: "Reasoning request in progress"
        case .speaking: "Voice response in progress"
        case .error: "Check the local diagnostics"
        case .offline: "Voice runtime is not connected"
        }
    }

    var symbolName: String {
        switch self {
        case .starting: "circle.dotted"
        case .idle: "waveform.circle"
        case .listening: "mic.fill"
        case .processing: "cpu"
        case .speaking: "waveform"
        case .error: "exclamationmark.triangle.fill"
        case .offline: "power"
        }
    }

    var isActive: Bool { self != .offline && self != .error }
}

struct VoiceStateEnvelope: Codable, Equatable, Sendable {
    let activityMilli: Int
    let generation: Int
    let state: CyrusRuntimeState
    let stateStartedAtUnixMs: Int64
    let updatedAtUnixMs: Int64
    let version: String

    enum CodingKeys: String, CodingKey {
        case activityMilli = "activity_milli"
        case generation
        case state
        case stateStartedAtUnixMs = "state_started_at_unix_ms"
        case updatedAtUnixMs = "updated_at_unix_ms"
        case version
    }

    static func decodeStrict(from data: Data) -> VoiceStateEnvelope? {
        guard let object = try? JSONSerialization.jsonObject(
            with: data,
            options: [.fragmentsAllowed]
        ) as? [String: Any],
        Set(object.keys) == Set([
            "activity_milli", "generation", "state",
            "state_started_at_unix_ms", "updated_at_unix_ms", "version",
        ])
        else { return nil }
        return try? JSONDecoder().decode(VoiceStateEnvelope.self, from: data)
    }

    func validated(nowUnixMs: Int64, staleAfterMs: Int64 = 3_500) -> CyrusRuntimeState? {
        guard version == "jarvis-voice-ui-state/2",
              (0...1_000).contains(activityMilli),
              generation > 0,
              stateStartedAtUnixMs > 0,
              stateStartedAtUnixMs <= updatedAtUnixMs,
              updatedAtUnixMs > 0,
              updatedAtUnixMs <= nowUnixMs + 5_000,
              nowUnixMs - updatedAtUnixMs <= staleAfterMs
        else { return nil }
        return state
    }

    var activity: Double { Double(activityMilli) / 1_000.0 }
}

enum PrivateSnapshotReader {
    static let maximumBytes = 512

    static func read(from url: URL) -> Data? {
        let descriptor = open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard descriptor >= 0 else { return nil }
        defer { close(descriptor) }

        var metadata = stat()
        guard fstat(descriptor, &metadata) == 0,
              metadata.st_uid == getuid(),
              (metadata.st_mode & S_IFMT) == S_IFREG,
              (metadata.st_mode & 0o777) == 0o600,
              metadata.st_size > 0,
              metadata.st_size <= maximumBytes
        else { return nil }

        var bytes = [UInt8](repeating: 0, count: Int(metadata.st_size))
        var offset = 0
        while offset < bytes.count {
            let remaining = bytes.count - offset
            let count = bytes.withUnsafeMutableBytes { buffer in
                Darwin.read(
                    descriptor,
                    buffer.baseAddress!.advanced(by: offset),
                    remaining
                )
            }
            guard count > 0 else { return nil }
            offset += count
        }
        return Data(bytes)
    }
}

enum CyrusUIArguments {
    static func stateFile(arguments: [String] = CommandLine.arguments) -> URL {
        if let index = arguments.firstIndex(of: "--state-file"),
           arguments.indices.contains(index + 1) {
            return URL(fileURLWithPath: arguments[index + 1])
        }
        return FileManager.default.homeDirectoryForCurrentUser
            // Compatibility boundary: Voice 0.8 still publishes to the frozen
            // legacy state root. The UI name can change without moving private
            // runtime state or invalidating existing beta installations.
            .appendingPathComponent("Library/Application Support/JARVIS Voice/state/ui-state.json")
    }

    static func demoEnabled(arguments: [String] = CommandLine.arguments) -> Bool {
        arguments.contains("--demo")
    }
}
