import Darwin
import AppKit
import Foundation
import SwiftUI
import Testing
@testable import CyrusUI

@Test func decodesCurrentStrictVoiceSnapshot() throws {
    let data = Data(#"{"activity_milli":640,"generation":7,"state":"listening","state_started_at_unix_ms":99000,"updated_at_unix_ms":100000,"version":"jarvis-voice-ui-state/2"}"#.utf8)
    let envelope = try JSONDecoder().decode(VoiceStateEnvelope.self, from: data)

    #expect(envelope.state == .listening)
    #expect(envelope.activity == 0.64)
    #expect(envelope.validated(nowUnixMs: 102_000) == .listening)
    #expect(envelope.validated(nowUnixMs: 104_000) == nil)
    #expect(VoiceStateEnvelope.decodeStrict(from: data) == envelope)

    let contentBearing = Data(#"{"activity_milli":640,"generation":7,"state":"listening","state_started_at_unix_ms":99000,"transcript":"private","updated_at_unix_ms":100000,"version":"jarvis-voice-ui-state/2"}"#.utf8)
    #expect(VoiceStateEnvelope.decodeStrict(from: contentBearing) == nil)
}

@Test func rejectsLegacyFutureAndInvalidSnapshots() throws {
    let legacy = VoiceStateEnvelope(
        activityMilli: 0,
        generation: 1,
        state: .idle,
        stateStartedAtUnixMs: 9_000,
        updatedAtUnixMs: 10_000,
        version: "jarvis-voice-ui-state/1"
    )
    let future = VoiceStateEnvelope(
        activityMilli: 0,
        generation: 1,
        state: .idle,
        stateStartedAtUnixMs: 19_000,
        updatedAtUnixMs: 20_000,
        version: "jarvis-voice-ui-state/2"
    )

    #expect(legacy.validated(nowUnixMs: 10_100) == nil)
    #expect(future.validated(nowUnixMs: 10_000) == nil)
}

@Test func privateReaderRejectsSymlinkAndLoosePermissions() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let snapshot = root.appendingPathComponent("ui-state.json")
    let outside = root.appendingPathComponent("outside.json")
    try Data("{}".utf8).write(to: outside)
    chmod(outside.path, 0o600)
    try FileManager.default.createSymbolicLink(at: snapshot, withDestinationURL: outside)

    #expect(PrivateSnapshotReader.read(from: snapshot) == nil)

    try FileManager.default.removeItem(at: snapshot)
    try Data("{}".utf8).write(to: snapshot)
    chmod(snapshot.path, 0o644)
    #expect(PrivateSnapshotReader.read(from: snapshot) == nil)
}

@MainActor
@Test func rendersCompleteHUDAtReferenceDesktopSize() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let snapshot = root.appendingPathComponent("ui-state.json")
    let now = Date(timeIntervalSince1970: 2_000)
    let fixtures: [(CyrusRuntimeState, Int)] = [
        (.listening, 720),
        (.processing, 0),
        (.speaking, 0),
        (.offline, 0),
    ]
    let renderDirectory = (
        ProcessInfo.processInfo.environment["CYRUS_UI_RENDER_DIRECTORY"]
            ?? ProcessInfo.processInfo.environment["JARVIS_UI_RENDER_DIRECTORY"]
    )
        .map { URL(fileURLWithPath: $0, isDirectory: true) }
    if let renderDirectory {
        try FileManager.default.createDirectory(
            at: renderDirectory,
            withIntermediateDirectories: true
        )
    }

    for (state, activity) in fixtures {
        let body = #"{"activity_milli":\#(activity),"generation":42,"state":"\#(state.rawValue)","state_started_at_unix_ms":1999000,"updated_at_unix_ms":2000000,"version":"jarvis-voice-ui-state/2"}"#
        try Data(body.utf8).write(to: snapshot, options: .atomic)
        chmod(snapshot.path, 0o600)
        let store = CyrusStateStore(
            stateURL: snapshot,
            now: { now },
            startTimer: false
        )
        let renderer = ImageRenderer(
            content: CyrusHUDView(store: store).frame(width: 980, height: 640)
        )
        renderer.scale = 2
        guard let image = renderer.nsImage,
              let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:])
        else {
            Issue.record("HUD failed to render state \(state.rawValue)")
            continue
        }
        #expect(bitmap.pixelsWide == 1_960)
        #expect(bitmap.pixelsHigh == 1_280)
        #expect(png.count > 100_000)
        if state == .listening,
           let output = (
               ProcessInfo.processInfo.environment["CYRUS_UI_RENDER_PATH"]
                   ?? ProcessInfo.processInfo.environment["JARVIS_UI_RENDER_PATH"]
           ) {
            try png.write(to: URL(fileURLWithPath: output), options: .atomic)
        }
        if let renderDirectory {
            try png.write(
                to: renderDirectory.appendingPathComponent("cyrus-ui-\(state.rawValue).png"),
                options: .atomic
            )
        }
    }
}
