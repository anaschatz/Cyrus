import AppKit
import SwiftUI

@main
struct JarvisUIApp: App {
    @StateObject private var store = JarvisStateStore()
    @AppStorage("keepHUDAboveWindows") private var keepHUDAboveWindows = false

    var body: some Scene {
        Window("JARVIS", id: "jarvis-hud") {
            JarvisHUDView(store: store)
                .frame(idealWidth: 980, idealHeight: 640)
                .background(WindowConfigurator(keepAbove: keepHUDAboveWindows))
        }
        .defaultSize(width: 980, height: 640)
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)

        MenuBarExtra {
            JarvisMenu(store: store, keepAbove: $keepHUDAboveWindows)
        } label: {
            Image(systemName: store.state.symbolName)
                .accessibilityLabel("JARVIS \(store.state.title)")
        }
        .menuBarExtraStyle(.menu)
    }
}

private struct JarvisMenu: View {
    @ObservedObject var store: JarvisStateStore
    @Binding var keepAbove: Bool
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Section {
            Label(store.state.title, systemImage: store.state.symbolName)
            Text(store.state.subtitle)
        }
        Divider()
        Button("Open JARVIS HUD") {
            openWindow(id: "jarvis-hud")
            NSApp.activate(ignoringOtherApps: true)
        }
        .keyboardShortcut("j")
        Button("Refresh Runtime State") { store.refresh() }
            .keyboardShortcut("r")
        Toggle("Keep HUD Above Windows", isOn: $keepAbove)
        Divider()
        Button("Quit JARVIS UI") { NSApp.terminate(nil) }
            .keyboardShortcut("q")
    }
}

private struct WindowConfigurator: NSViewRepresentable {
    let keepAbove: Bool

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { configure(view.window) }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async { configure(nsView.window) }
    }

    private func configure(_ window: NSWindow?) {
        guard let window else { return }
        window.isOpaque = false
        window.backgroundColor = .clear
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.collectionBehavior.insert(.fullScreenAuxiliary)
        window.level = keepAbove ? .floating : .normal
    }
}
