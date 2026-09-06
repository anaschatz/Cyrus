import Foundation
import Testing
@testable import JarvisActionNative

@Test func numericInputsDoNotTruncateOrCoerceBooleans() throws {
    #expect(try integer(["limit": 5], "limit") == 5)
    #expect(throws: (any Error).self) { try integer(["limit": 1.5], "limit") }
    #expect(throws: (any Error).self) { try integer(["limit": true], "limit") }
    #expect(try bool(["has_due": true], "has_due"))
    #expect(throws: (any Error).self) { try bool(["has_due": 1], "has_due") }
}

@Test func reminderComponentsResolveWithoutAnEmbeddedCalendar() throws {
    var components = DateComponents()
    components.year = 2026
    components.month = 9
    components.day = 5
    components.hour = 10
    components.minute = 30
    components.second = 0
    components.nanosecond = 123_000_000
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    let date = try #require(reminderDate(components, fallbackTimezone: "UTC"))
    #expect(formatter.string(from: date) == "2026-09-05T10:30:00.123Z")
    components.timeZone = TimeZone(secondsFromGMT: 3 * 3600)
    let zoned = try #require(reminderDate(components, fallbackTimezone: "UTC"))
    #expect(formatter.string(from: zoned) == "2026-09-05T07:30:00.123Z")
}
