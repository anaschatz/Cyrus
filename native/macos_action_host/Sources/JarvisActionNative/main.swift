import AppKit
import CryptoKit
import EventKit
import Foundation
import UniformTypeIdentifiers

enum NativeError: Error {
    case safe(String)
}

final class OpenResultBox: @unchecked Sendable {
    var error: Error?
}

final class MetadataQueryBox: @unchecked Sendable {
    var finished = false
}

let utcFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.calendar = Calendar(identifier: .gregorian)
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
    return formatter
}()

func utcText(_ date: Date = Date()) -> String { utcFormatter.string(from: date) }

func sha256(_ value: Data) -> String {
    SHA256.hash(data: value).map { String(format: "%02x", $0) }.joined()
}

func exactDictionary(_ value: Any, fields: Set<String>) throws -> [String: Any] {
    guard let row = value as? [String: Any], Set(row.keys) == fields else {
        throw NativeError.safe("host.adapter_rejected")
    }
    return row
}

func string(_ row: [String: Any], _ name: String) throws -> String {
    guard let value = row[name] as? String else { throw NativeError.safe("host.adapter_rejected") }
    return value
}

func bool(_ row: [String: Any], _ name: String) throws -> Bool {
    guard let value = row[name] as? Bool else { throw NativeError.safe("host.adapter_rejected") }
    return value
}

func integer(_ row: [String: Any], _ name: String) throws -> Int {
    guard let value = row[name] as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID() else {
        throw NativeError.safe("host.adapter_rejected")
    }
    return value.intValue
}

func localDate(_ value: String, timezone: String) throws -> Date {
    guard let zone = TimeZone(identifier: timezone) else { throw NativeError.safe("calendar.time_nonexistent") }
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.calendar = Calendar(identifier: .gregorian)
    formatter.timeZone = zone
    formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS"
    formatter.isLenient = false
    guard let date = formatter.date(from: value), formatter.string(from: date) == value else {
        throw NativeError.safe("calendar.time_nonexistent")
    }
    return date
}

func localText(_ date: Date, timezone: String) throws -> String {
    guard let zone = TimeZone(identifier: timezone) else { throw NativeError.safe("calendar.time_nonexistent") }
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.calendar = Calendar(identifier: .gregorian)
    formatter.timeZone = zone
    formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS"
    return formatter.string(from: date)
}

func stableReference(_ domain: String, _ identifier: String) -> String {
    let digest = Array(SHA256.hash(data: Data((domain + "\u{0}" + identifier).utf8)).prefix(16))
    var bytes = digest
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    let uuid = UUID(uuid: (
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    ))
    return uuid.uuidString.lowercased()
}

func eventAuthorization(read: Bool) throws -> String {
    let status = EKEventStore.authorizationStatus(for: .event)
    if #available(macOS 14.0, *) {
        if status == .fullAccess { return "FULL_ACCESS" }
        if !read && status == .writeOnly { return "WRITE_ONLY" }
    }
    throw NativeError.safe(read ? "calendar.full_access_required" : "calendar.permission_revoked")
}

func reminderAuthorization() throws {
    let status = EKEventStore.authorizationStatus(for: .reminder)
    if #available(macOS 14.0, *), status == .fullAccess { return }
    throw NativeError.safe("reminder.full_access_required")
}

func calendarRead(arguments: [String: Any]) throws -> ([String: Any], [[String: Any]]) {
    _ = try eventAuthorization(read: true)
    let timezone = try string(arguments, "timezone")
    let start = try localDate(string(arguments, "start_local"), timezone: timezone)
    let end = try localDate(string(arguments, "end_local"), timezone: timezone)
    let limit = try integer(arguments, "limit")
    let offset = (arguments["_cursor_offset"] as? NSNumber)?.intValue ?? 0
    let store = EKEventStore()
    let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
    let selected = store.events(matching: predicate).sorted {
        if $0.startDate != $1.startDate { return $0.startDate < $1.startDate }
        if $0.endDate != $1.endDate { return $0.endDate < $1.endDate }
        return ($0.eventIdentifier ?? "") < ($1.eventIdentifier ?? "")
    }
    let page = Array(selected.dropFirst(offset).prefix(limit))
    var bindings: [[String: Any]] = []
    let events: [[String: Any]] = try page.map { event in
        guard let identifier = event.eventIdentifier else { throw NativeError.safe("calendar.event_missing_after_write") }
        let reference = stableReference("calendar", identifier)
        bindings.append(["ref": reference, "kind": "calendar", "external_id": identifier])
        let eventTimezone = event.timeZone?.identifier ?? timezone
        return [
            "event_ref": reference,
            "calendar_label": event.calendar.title,
            "title": event.title ?? "",
            "start_local": try localText(event.startDate, timezone: eventTimezone),
            "end_local": try localText(event.endDate, timezone: eventTimezone),
            "start_utc": utcText(event.startDate),
            "end_utc": utcText(event.endDate),
            "timezone": eventTimezone,
            "all_day": event.isAllDay,
            "location": event.location ?? "",
            "notes": event.notes ?? "",
        ]
    }
    return ([
        "observed_at_utc": utcText(),
        "events": events,
        "next_cursor": "",
        "truncated": offset + page.count < selected.count,
        "_next_offset": offset + page.count,
    ], bindings)
}

func calendarCreate(arguments: [String: Any]) throws -> ([String: Any], [[String: Any]], String, String) {
    let authorization = try eventAuthorization(read: false)
    let timezone = try string(arguments, "timezone")
    let store = EKEventStore()
    let event = EKEvent(eventStore: store)
    event.calendar = store.defaultCalendarForNewEvents
    guard event.calendar != nil else { throw NativeError.safe("calendar.permission_revoked") }
    event.title = try string(arguments, "title")
    event.startDate = try localDate(string(arguments, "start_local"), timezone: timezone)
    event.endDate = try localDate(string(arguments, "end_local"), timezone: timezone)
    event.timeZone = TimeZone(identifier: timezone)
    event.location = try string(arguments, "location")
    event.notes = try string(arguments, "notes")
    event.availability = try string(arguments, "availability") == "FREE" ? .free : .busy
    try store.save(event, span: .thisEvent, commit: true)
    guard let identifier = event.eventIdentifier else { throw NativeError.safe("calendar.event_missing_after_write") }
    let reference = stableReference("calendar", identifier)
    var verification = "UNAVAILABLE"
    var status = "ACCEPTED"
    var reason = "calendar.write_only_no_readback"
    if authorization == "FULL_ACCESS" {
        guard let readback = store.event(withIdentifier: identifier),
              readback.title == event.title,
              readback.startDate == event.startDate,
              readback.endDate == event.endDate,
              (readback.location ?? "") == (event.location ?? ""),
              (readback.notes ?? "") == (event.notes ?? ""),
              readback.availability == event.availability else {
            throw NativeError.safe("calendar.event_missing_after_write")
        }
        verification = "EXACT_READBACK"
        status = "VERIFIED"
        reason = "ok"
    }
    let notes = try string(arguments, "notes")
    return ([
        "event_ref": reference,
        "authorization_mode": authorization,
        "title": try string(arguments, "title"),
        "start_local": try string(arguments, "start_local"),
        "end_local": try string(arguments, "end_local"),
        "timezone": timezone,
        "location": try string(arguments, "location"),
        "notes_sha256": sha256(Data(notes.utf8)),
        "availability": try string(arguments, "availability"),
    ], [["ref": reference, "kind": "calendar", "external_id": identifier]], status, verification + "|" + reason)
}

func fetchReminders(_ predicate: NSPredicate) throws -> [EKReminder] {
    let store = EKEventStore()
    let semaphore = DispatchSemaphore(value: 0)
    var selected: [EKReminder]?
    store.fetchReminders(matching: predicate) { reminders in
        selected = reminders
        semaphore.signal()
    }
    guard semaphore.wait(timeout: .now() + 10) == .success, let result = selected else {
        throw NativeError.safe("host.deadline_expired")
    }
    return result
}

func reminderItem(_ reminder: EKReminder, fallbackTimezone: String) throws -> ([String: Any], [String: Any]) {
    guard let identifier = reminder.calendarItemIdentifier as String? else { throw NativeError.safe("reminder.not_found") }
    let reference = stableReference("reminder", identifier)
    var dueLocal = ""
    var dueUTC = ""
    var timezone = ""
    if let components = reminder.dueDateComponents, let date = components.date {
        timezone = components.timeZone?.identifier ?? fallbackTimezone
        dueLocal = try localText(date, timezone: timezone)
        dueUTC = utcText(date)
    }
    let priority: String
    switch reminder.priority {
    case 1...4: priority = "HIGH"
    case 5: priority = "MEDIUM"
    case 6...9: priority = "LOW"
    default: priority = "NONE"
    }
    return ([
        "reminder_ref": reference,
        "title": reminder.title ?? "",
        "notes": reminder.notes ?? "",
        "has_due": reminder.dueDateComponents != nil,
        "due_local": dueLocal,
        "due_utc": dueUTC,
        "timezone": timezone,
        "is_completed": reminder.isCompleted,
        "priority": priority,
    ], ["ref": reference, "kind": "reminder", "external_id": identifier])
}

func reminderList(arguments: [String: Any]) throws -> ([String: Any], [[String: Any]]) {
    try reminderAuthorization()
    let store = EKEventStore()
    let completed = try string(arguments, "status") == "COMPLETED"
    let predicate = store.predicateForReminders(in: nil)
    var windowStart: Date?
    var windowEnd: Date?
    if try string(arguments, "due_filter") == "WINDOW" {
        let timezone = try string(arguments, "timezone")
        windowStart = try localDate(string(arguments, "due_from_local"), timezone: timezone)
        windowEnd = try localDate(string(arguments, "due_to_local"), timezone: timezone)
    }
    let all = try fetchReminders(predicate).filter {
        guard $0.isCompleted == completed else { return false }
        guard let start = windowStart, let end = windowEnd else { return true }
        guard let due = $0.dueDateComponents?.date else { return false }
        return due >= start && due < end
    }
    let timezone = (arguments["timezone"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "UTC"
    var mapped = try all.map { try reminderItem($0, fallbackTimezone: timezone) }
    mapped.sort {
        let left = ($0.0["due_utc"] as? String) ?? ""
        let right = ($1.0["due_utc"] as? String) ?? ""
        if left.isEmpty != right.isEmpty { return !left.isEmpty }
        if left != right { return left < right }
        return (($0.0["title"] as? String) ?? "").localizedCaseInsensitiveCompare(($1.0["title"] as? String) ?? "") == .orderedAscending
    }
    let limit = try integer(arguments, "limit")
    let offset = (arguments["_cursor_offset"] as? NSNumber)?.intValue ?? 0
    let page = Array(mapped.dropFirst(offset).prefix(limit))
    return ([
        "observed_at_utc": utcText(),
        "reminders": page.map { $0.0 },
        "next_cursor": "",
        "truncated": offset + page.count < mapped.count,
        "_next_offset": offset + page.count,
    ], page.map { $0.1 })
}

func reminderCreate(arguments: [String: Any]) throws -> ([String: Any], [[String: Any]]) {
    try reminderAuthorization()
    let store = EKEventStore()
    let reminder = EKReminder(eventStore: store)
    reminder.calendar = store.defaultCalendarForNewReminders()
    guard reminder.calendar != nil else { throw NativeError.safe("reminder.permission_revoked") }
    reminder.title = try string(arguments, "title")
    reminder.notes = try string(arguments, "notes")
    let hasDue = try bool(arguments, "has_due")
    var dueLocal = ""
    var dueUTC = ""
    var timezone = ""
    if hasDue {
        timezone = try string(arguments, "timezone")
        dueLocal = try string(arguments, "due_local")
        let date = try localDate(dueLocal, timezone: timezone)
        dueUTC = utcText(date)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: timezone)!
        reminder.dueDateComponents = calendar.dateComponents(
            [.year, .month, .day, .hour, .minute, .second, .timeZone], from: date
        )
    }
    switch try string(arguments, "priority") {
    case "HIGH": reminder.priority = 1
    case "MEDIUM": reminder.priority = 5
    case "LOW": reminder.priority = 9
    default: reminder.priority = 0
    }
    try store.save(reminder, commit: true)
    let identifier = reminder.calendarItemIdentifier
    guard let readback = store.calendarItem(withIdentifier: identifier) as? EKReminder,
          readback.title == reminder.title,
          (readback.notes ?? "") == (reminder.notes ?? ""),
          readback.priority == reminder.priority,
          readback.dueDateComponents?.date == reminder.dueDateComponents?.date else {
        throw NativeError.safe("reminder.not_found")
    }
    let reference = stableReference("reminder", identifier)
    let notes = try string(arguments, "notes")
    return ([
        "reminder_ref": reference,
        "title": try string(arguments, "title"),
        "notes_sha256": sha256(Data(notes.utf8)),
        "has_due": hasDue,
        "due_local": dueLocal,
        "due_utc": dueUTC,
        "timezone": timezone,
        "priority": try string(arguments, "priority"),
    ], [["ref": reference, "kind": "reminder", "external_id": identifier]])
}

func reminderComplete(arguments: [String: Any], context: [String: Any]) throws -> ([String: Any], [[String: Any]]) {
    try reminderAuthorization()
    guard let binding = context["reference_binding"] as? [String: Any],
          binding["kind"] as? String == "reminder",
          let identifier = binding["external_id"] as? String else {
        throw NativeError.safe("reminder.ref_invalid")
    }
    let store = EKEventStore()
    guard let reminder = store.calendarItem(withIdentifier: identifier) as? EKReminder else {
        throw NativeError.safe("reminder.not_found")
    }
    reminder.isCompleted = true
    reminder.completionDate = Date()
    try store.save(reminder, commit: true)
    guard let readback = store.calendarItem(withIdentifier: identifier) as? EKReminder, readback.isCompleted else {
        throw NativeError.safe("reminder.not_found")
    }
    return ([
        "reminder_ref": try string(arguments, "reminder_ref"),
        "completed_at_utc": utcText(readback.completionDate ?? Date()),
    ], [])
}

let qualifiedApps: [String: (bundle: String, name: String)] = [
    "finder": ("com.apple.finder", "Finder"),
    "preview": ("com.apple.Preview", "Preview"),
    "textedit": ("com.apple.TextEdit", "TextEdit"),
    "calendar": ("com.apple.iCal", "Calendar"),
    "reminders": ("com.apple.reminders", "Reminders"),
    "mail": ("com.apple.mail", "Mail"),
]

func appOpen(arguments: [String: Any]) throws -> [String: Any] {
    let appID = try string(arguments, "app_id")
    guard let qualified = qualifiedApps[appID],
          let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: qualified.bundle) else {
        throw NativeError.safe("file.handler_denied")
    }
    let configuration = NSWorkspace.OpenConfiguration()
    configuration.activates = true
    let semaphore = DispatchSemaphore(value: 0)
    let result = OpenResultBox()
    NSWorkspace.shared.openApplication(at: url, configuration: configuration) { _, error in
        result.error = error
        semaphore.signal()
    }
    guard semaphore.wait(timeout: .now() + 10) == .success, result.error == nil else {
        throw NativeError.safe("host.adapter_rejected")
    }
    return ["app_id": appID, "bundle_id": qualified.bundle, "request_observed_at_utc": utcText()]
}

func fileSearch(arguments: [String: Any], context: [String: Any]) throws -> ([String: Any], [[String: Any]]) {
    guard let roots = context["roots"] as? [String: String],
          let selectedRoots = arguments["root_ids"] as? [String],
          let kinds = arguments["kinds"] as? [String] else {
        throw NativeError.safe("file.root_unavailable")
    }
    let query = try string(arguments, "query").folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
    let offset = (arguments["_cursor_offset"] as? NSNumber)?.intValue ?? 0
    let limit = try integer(arguments, "limit")
    var matches: [(path: String, root: String, values: URLResourceValues)] = []
    let keys: Set<URLResourceKey> = [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey, .contentTypeKey, .fileResourceIdentifierKey, .volumeIdentifierKey]
    var scopedRoots: [(id: String, url: URL)] = []
    for rootID in selectedRoots {
        guard let rootPath = roots[rootID] else { throw NativeError.safe("file.root_unavailable") }
        let rootURL = URL(fileURLWithPath: rootPath).standardizedFileURL.resolvingSymlinksInPath()
        guard FileManager.default.fileExists(atPath: rootURL.path) else { throw NativeError.safe("file.root_unavailable") }
        scopedRoots.append((rootID, rootURL))
    }
    let metadata = NSMetadataQuery()
    metadata.searchScopes = scopedRoots.map { $0.url.path }
    metadata.predicate = NSPredicate(
        format: "(%K CONTAINS[cd] %@) OR (%K CONTAINS[cd] %@)",
        NSMetadataItemFSNameKey, query, NSMetadataItemPathKey, query
    )
    let queryBox = MetadataQueryBox()
    let observer = NotificationCenter.default.addObserver(
        forName: .NSMetadataQueryDidFinishGathering,
        object: metadata,
        queue: nil
    ) { _ in queryBox.finished = true }
    defer { NotificationCenter.default.removeObserver(observer); metadata.stop() }
    guard metadata.start() else { throw NativeError.safe("file.root_unavailable") }
    let gatheringDeadline = Date().addingTimeInterval(5)
    while !queryBox.finished && Date() < gatheringDeadline {
        _ = RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    guard queryBox.finished else { throw NativeError.safe("host.deadline_expired") }
    metadata.disableUpdates()
    for case let item as NSMetadataItem in metadata.results {
        guard let rawPath = item.value(forAttribute: NSMetadataItemPathKey) as? String else { continue }
        let rawURL = URL(fileURLWithPath: rawPath).standardizedFileURL
        let preliminary = try rawURL.resourceValues(forKeys: [.isSymbolicLinkKey])
        if preliminary.isSymbolicLink == true { continue }
        let candidate = rawURL.resolvingSymlinksInPath()
        guard let selectedRoot = scopedRoots.first(where: { candidate.path.hasPrefix($0.url.path + "/") }) else { continue }
        let values = try candidate.resourceValues(forKeys: keys)
        let kind = values.isDirectory == true ? "DIRECTORY" : "FILE"
        if !kinds.contains(kind) { continue }
        let relative = String(candidate.path.dropFirst(selectedRoot.url.path.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let folded = relative.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
        if folded.contains(query) { matches.append((relative, selectedRoot.id, values)) }
        if matches.count > 10_000 { throw NativeError.safe("tool.payload_too_large") }
    }
    matches.sort {
        let l = $0.path.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
        let r = $1.path.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
        if l != r { return l < r }
        let leftBytes = Array($0.path.utf8)
        let rightBytes = Array($1.path.utf8)
        if leftBytes != rightBytes { return leftBytes.lexicographicallyPrecedes(rightBytes) }
        if $0.root != $1.root { return $0.root < $1.root }
        let leftIdentity = $0.values.fileResourceIdentifier.map { String(describing: $0) } ?? ""
        let rightIdentity = $1.values.fileResourceIdentifier.map { String(describing: $0) } ?? ""
        return leftIdentity < rightIdentity
    }
    let page = Array(matches.dropFirst(offset).prefix(limit))
    var bindings: [[String: Any]] = []
    let expires = Date().addingTimeInterval(600)
    let items: [[String: Any]] = page.map { item in
        let identity = item.values.fileResourceIdentifier.map { String(describing: $0) } ?? ""
        let volume = item.values.volumeIdentifier.map { String(describing: $0) } ?? ""
        let reference = stableReference("file", item.root + "\u{0}" + item.path + "\u{0}" + identity)
        bindings.append([
            "ref": reference, "kind": "file", "root_id": item.root,
            "relative_path": item.path, "identity": identity, "volume": volume,
            "expires_at_utc": utcText(expires),
        ])
        let name = (item.path as NSString).lastPathComponent
        let parent = (item.path as NSString).deletingLastPathComponent
        return [
            "file_ref": reference, "root_id": item.root, "display_name": name,
            "relative_parent": parent == "." ? "" : parent, "kind": item.values.isDirectory == true ? "DIRECTORY" : "FILE",
            "uti": item.values.contentType?.identifier ?? "public.data",
            "size_bytes": item.values.fileSize ?? 0,
            "modified_at_utc": utcText(item.values.contentModificationDate ?? Date(timeIntervalSince1970: 0)),
            "ref_expires_at_utc": utcText(expires),
        ]
    }
    return ([
        "observed_at_utc": utcText(), "items": items, "next_cursor": "",
        "truncated": offset + page.count < matches.count, "_next_offset": offset + page.count,
    ], bindings)
}

func fileOpen(arguments: [String: Any], context: [String: Any]) throws -> [String: Any] {
    guard let roots = context["roots"] as? [String: String],
          let binding = context["reference_binding"] as? [String: Any],
          binding["kind"] as? String == "file",
          let rootID = binding["root_id"] as? String,
          let root = roots[rootID],
          let relative = binding["relative_path"] as? String,
          let expiresText = binding["expires_at_utc"] as? String,
          let expires = utcFormatter.date(from: expiresText), expires > Date() else {
        throw NativeError.safe("file.ref_expired")
    }
    let rootURL = URL(fileURLWithPath: root).standardizedFileURL.resolvingSymlinksInPath()
    let url = rootURL.appendingPathComponent(relative).standardizedFileURL.resolvingSymlinksInPath()
    guard url.path.hasPrefix(rootURL.path + "/") else { throw NativeError.safe("file.outside_root") }
    let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey, .fileResourceIdentifierKey, .volumeIdentifierKey, .contentTypeKey])
    let identity = values.fileResourceIdentifier.map { String(describing: $0) } ?? ""
    let volume = values.volumeIdentifier.map { String(describing: $0) } ?? ""
    guard values.isSymbolicLink != true,
          identity == (binding["identity"] as? String),
          volume == (binding["volume"] as? String) else {
        throw NativeError.safe("file.identity_changed")
    }
    let allowed = [UTType.directory, UTType.plainText, UTType.image, UTType.pdf, UTType.json, UTType.xml, UTType.commaSeparatedText]
    guard let type = values.contentType, allowed.contains(where: { type.conforms(to: $0) }) else {
        throw NativeError.safe("file.type_denied")
    }
    let handlerID: String
    if values.isDirectory == true { handlerID = "finder" }
    else if type.conforms(to: .image) || type.conforms(to: .pdf) { handlerID = "preview" }
    else { handlerID = "textedit" }
    guard let qualified = qualifiedApps[handlerID],
          let application = NSWorkspace.shared.urlForApplication(withBundleIdentifier: qualified.bundle) else {
        throw NativeError.safe("file.handler_denied")
    }
    let configuration = NSWorkspace.OpenConfiguration()
    configuration.activates = true
    let semaphore = DispatchSemaphore(value: 0)
    let openResult = OpenResultBox()
    NSWorkspace.shared.open([url], withApplicationAt: application, configuration: configuration) { _, error in
        openResult.error = error
        semaphore.signal()
    }
    guard semaphore.wait(timeout: .now() + 10) == .success, openResult.error == nil else {
        throw NativeError.safe("host.adapter_rejected")
    }
    return [
        "file_ref": try string(arguments, "file_ref"),
        "handler_id": handlerID,
        "request_observed_at_utc": utcText(),
    ]
}

func appleScriptLiteral(_ value: String) -> String {
    let pieces = value.split(separator: "\n", omittingEmptySubsequences: false).map {
        "\"" + $0.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
    }
    return pieces.joined(separator: " & linefeed & ")
}

func emailDraft(arguments: [String: Any]) throws -> ([String: Any], [[String: Any]]) {
    guard let recipients = arguments["to"] as? [String] else { throw NativeError.safe("email.address_invalid") }
    let subject = try string(arguments, "subject")
    let body = try string(arguments, "body")
    let recipientCommands = recipients.map {
        "make new to recipient at end of to recipients with properties {address:" + appleScriptLiteral($0) + "}"
    }.joined(separator: "\n")
    let source = """
    tell application "Mail"
      set m to make new outgoing message with properties {subject:\(appleScriptLiteral(subject)), content:\(appleScriptLiteral(body)), visible:false}
      tell m
        \(recipientCommands)
      end tell
      save m
      set actualRecipients to {}
      repeat with r in (to recipients of m)
        set end of actualRecipients to (address of r) as string
      end repeat
      return {(id of m) as string, (subject of m) as string, (content of m) as string, actualRecipients}
    end tell
    """
    var error: NSDictionary?
    guard let descriptor = NSAppleScript(source: source)?.executeAndReturnError(&error), error == nil else {
        throw NativeError.safe("email.permission_revoked")
    }
    guard descriptor.numberOfItems == 4,
          let identifier = descriptor.atIndex(1)?.stringValue,
          let readbackSubject = descriptor.atIndex(2)?.stringValue,
          let readbackBody = descriptor.atIndex(3)?.stringValue,
          let recipientDescriptor = descriptor.atIndex(4),
          !identifier.isEmpty else { throw NativeError.safe("email.draft_receipt_missing") }
    var readbackRecipients: [String] = []
    if recipientDescriptor.numberOfItems > 0 {
        for index in 1...recipientDescriptor.numberOfItems {
            guard let address = recipientDescriptor.atIndex(index)?.stringValue else {
                throw NativeError.safe("email.draft_receipt_missing")
            }
            readbackRecipients.append(address)
        }
    }
    guard readbackSubject == subject,
          readbackBody == body,
          readbackRecipients == recipients else {
        throw NativeError.safe("email.draft_receipt_missing")
    }
    let reference = stableReference("mail-draft", identifier)
    return ([
        "draft_ref": reference,
        "to": recipients,
        "subject": subject,
        "body_sha256": sha256(Data(body.utf8)),
        "created_at_utc": utcText(),
    ], [["ref": reference, "kind": "mail_draft", "external_id": identifier]])
}

func run() throws -> [String: Any] {
    let input = FileHandle.standardInput.readDataToEndOfFile()
    guard input.count > 0, input.count <= 262_144 else { throw NativeError.safe("tool.payload_too_large") }
    let decoded = try JSONSerialization.jsonObject(with: input, options: [])
    let request = try exactDictionary(decoded, fields: ["tool_id", "arguments", "context"])
    let toolID = try string(request, "tool_id")
    guard let arguments = request["arguments"] as? [String: Any], let context = request["context"] as? [String: Any] else {
        throw NativeError.safe("host.adapter_rejected")
    }
    var result: [String: Any]
    var bindings: [[String: Any]] = []
    var status = "VERIFIED"
    var verification = "EXACT_READBACK"
    var reason = "ok"
    switch toolID {
    case "app.open@1":
        result = try appOpen(arguments: arguments)
        status = "ACCEPTED"; verification = "UNAVAILABLE"; reason = "tool.verification_unavailable"
    case "file.search@1": (result, bindings) = try fileSearch(arguments: arguments, context: context)
    case "file.open@1":
        result = try fileOpen(arguments: arguments, context: context)
        status = "ACCEPTED"; verification = "UNAVAILABLE"; reason = "tool.verification_unavailable"
    case "calendar.read@1": (result, bindings) = try calendarRead(arguments: arguments)
    case "calendar.create@1":
        let created = try calendarCreate(arguments: arguments)
        result = created.0; bindings = created.1; status = created.2
        let pieces = created.3.split(separator: "|", maxSplits: 1).map(String.init)
        verification = pieces[0]; reason = pieces[1]
    case "reminder.list@1": (result, bindings) = try reminderList(arguments: arguments)
    case "reminder.create@1": (result, bindings) = try reminderCreate(arguments: arguments)
    case "reminder.complete@1": (result, bindings) = try reminderComplete(arguments: arguments, context: context)
    case "email.draft@1": (result, bindings) = try emailDraft(arguments: arguments)
    default: throw NativeError.safe("tool.unknown")
    }
    return [
        "status": status, "verification": verification, "reason_code": reason,
        "result_payload": result, "reference_bindings": bindings,
        "os_receipt_sha256": sha256(Data((toolID + "\u{0}" + utcText()).utf8)),
    ]
}

do {
    let output = try run()
    let encoded = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys, .withoutEscapingSlashes])
    FileHandle.standardOutput.write(encoded)
} catch NativeError.safe(let code) {
    let encoded = try! JSONSerialization.data(withJSONObject: ["error": code], options: [.sortedKeys])
    FileHandle.standardOutput.write(encoded)
    exit(2)
} catch {
    let encoded = try! JSONSerialization.data(withJSONObject: ["error": "host.internal_safe_failure"], options: [.sortedKeys])
    FileHandle.standardOutput.write(encoded)
    exit(3)
}
