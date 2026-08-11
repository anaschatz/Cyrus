import AppKit
import CryptoKit
import Foundation
@preconcurrency import UserNotifications

let callbackContract = "jarvis.scheduler-native-callback/1"
let categoryIdentifier = "JARVIS_SCHEDULER_ACTIONS_V1"
let snoozeIdentifier = "JARVIS_SNOOZE_10_MINUTES"
let cancelIdentifier = "JARVIS_CANCEL"
let maximumInputBytes = 32_768

let utcFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.calendar = Calendar(identifier: .gregorian)
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
    return formatter
}()

func utcText(_ date: Date = Date()) -> String { utcFormatter.string(from: date) }

final class OutputWriter: @unchecked Sendable {
    private let lock = NSLock()

    func write(_ value: [String: Any]) {
        guard let data = try? JSONSerialization.data(
            withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]
        ) else { return }
        lock.lock()
        defer { lock.unlock() }
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0a]))
    }
}

final class CallbackSigner: @unchecked Sendable {
    private let lock = NSLock()
    private var key: SymmetricKey?

    func initialize(_ raw: Data) -> Bool {
        guard raw.count == 32 else { return false }
        lock.lock()
        defer { lock.unlock() }
        guard key == nil else { return false }
        key = SymmetricKey(data: raw)
        return true
    }

    func sign(_ core: [String: Any]) -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let key,
              let encoded = try? JSONSerialization.data(
                withJSONObject: core, options: [.sortedKeys, .withoutEscapingSlashes]
              ) else { return nil }
        let code = HMAC<SHA256>.authenticationCode(for: encoded, using: key)
        return Data(code).map { String(format: "%02x", $0) }.joined()
    }
}

final class SchedulerNotificationDelegate: NSObject, UNUserNotificationCenterDelegate, @unchecked Sendable {
    let output: OutputWriter
    let signer: CallbackSigner

    init(output: OutputWriter, signer: CallbackSigner) {
        self.output = output
        self.signer = signer
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .list, .sound])
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        defer { completionHandler() }
        let action: String
        let requestedDue: Any
        switch response.actionIdentifier {
        case snoozeIdentifier:
            action = "SNOOZE"
            requestedDue = utcText(Date().addingTimeInterval(600))
        case cancelIdentifier:
            action = "CANCEL"
            requestedDue = NSNull()
        default:
            return
        }
        let identifier = response.notification.request.identifier
        guard identifier.count == 64,
              identifier.allSatisfy({ $0.isHexDigit && !$0.isUppercase }) else { return }
        var core: [String: Any] = [
            "contract": callbackContract,
            "callback_id": UUID().uuidString.lowercased(),
            "notification_id": identifier,
            "action": action,
            "requested_due_at_utc": requestedDue,
            "occurred_at_utc": utcText(),
        ]
        guard let mac = signer.sign(core) else { return }
        core["mac"] = mac
        output.write(core)
    }
}

func exactObject(_ value: Any, fields: Set<String>) -> [String: Any]? {
    guard let row = value as? [String: Any], Set(row.keys) == fields else { return nil }
    return row
}

func receipt(_ output: OutputWriter, requestID: String, accepted: Bool, reason: String, ambiguous: Bool = false) {
    output.write([
        "type": "command.receipt",
        "request_id": requestID,
        "accepted": accepted,
        "ambiguous": ambiguous,
        "reason_code": reason,
    ])
}

@MainActor func handleCommand(
    _ value: Any,
    center: UNUserNotificationCenter,
    signer: CallbackSigner,
    output: OutputWriter
) {
    guard let preview = value as? [String: Any], let type = preview["type"] as? String else {
        receipt(output, requestID: "00000000-0000-4000-8000-000000000000", accepted: false, reason: "invalid_command")
        return
    }
    if type == "initialize" {
        guard let row = exactObject(preview, fields: ["type", "request_id", "callback_key"]),
              let requestID = row["request_id"] as? String,
              let encoded = row["callback_key"] as? String,
              let key = Data(base64Encoded: encoded), signer.initialize(key) else {
            receipt(output, requestID: (preview["request_id"] as? String) ?? "00000000-0000-4000-8000-000000000000", accepted: false, reason: "initialization_rejected")
            return
        }
        receipt(output, requestID: requestID, accepted: true, reason: "initialized")
        return
    }
    if type == "remove" {
        guard let row = exactObject(preview, fields: ["type", "request_id", "notification_id"]),
              let requestID = row["request_id"] as? String,
              let identifier = row["notification_id"] as? String else {
            receipt(output, requestID: "00000000-0000-4000-8000-000000000000", accepted: false, reason: "invalid_command")
            return
        }
        center.removeDeliveredNotifications(withIdentifiers: [identifier])
        center.removePendingNotificationRequests(withIdentifiers: [identifier])
        receipt(output, requestID: requestID, accepted: true, reason: "native_api_accepted")
        return
    }
    guard type == "post",
          let row = exactObject(preview, fields: ["type", "request_id", "notification_id", "title", "body", "silent"]),
          let requestID = row["request_id"] as? String,
          let identifier = row["notification_id"] as? String,
          let title = row["title"] as? String,
          let body = row["body"] as? String,
          let silent = row["silent"] as? Bool else {
        receipt(output, requestID: (preview["request_id"] as? String) ?? "00000000-0000-4000-8000-000000000000", accepted: false, reason: "invalid_command")
        return
    }
    center.getNotificationSettings { settings in
        guard settings.authorizationStatus == .authorized || settings.authorizationStatus == .provisional else {
            receipt(output, requestID: requestID, accepted: false, reason: "permission_denied")
            return
        }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.categoryIdentifier = categoryIdentifier
        content.sound = silent ? nil : .default
        center.add(UNNotificationRequest(identifier: identifier, content: content, trigger: nil)) { error in
            receipt(
                output,
                requestID: requestID,
                accepted: error == nil,
                reason: error == nil ? "native_api_accepted" : "native_api_rejected",
                ambiguous: false
            )
        }
    }
}

let output = OutputWriter()
let signer = CallbackSigner()
let center = UNUserNotificationCenter.current()
let delegate = SchedulerNotificationDelegate(output: output, signer: signer)
center.delegate = delegate
center.setNotificationCategories([
    UNNotificationCategory(
        identifier: categoryIdentifier,
        actions: [
            UNNotificationAction(identifier: snoozeIdentifier, title: "Snooze 10 minutes"),
            UNNotificationAction(identifier: cancelIdentifier, title: "Cancel", options: [.destructive]),
        ],
        intentIdentifiers: [],
        options: []
    )
])

Thread.detachNewThread {
    while let line = readLine(strippingNewline: true) {
        guard let data = line.data(using: .utf8), data.count <= maximumInputBytes else {
            receipt(output, requestID: "00000000-0000-4000-8000-000000000000", accepted: false, reason: "invalid_command")
            continue
        }
        DispatchQueue.main.async {
            guard let value = try? JSONSerialization.jsonObject(with: data) else {
                receipt(output, requestID: "00000000-0000-4000-8000-000000000000", accepted: false, reason: "invalid_command")
                return
            }
            handleCommand(value, center: center, signer: signer, output: output)
        }
    }
    DispatchQueue.main.async { NSApplication.shared.terminate(nil) }
}

let application = NSApplication.shared
application.setActivationPolicy(.accessory)
application.run()
