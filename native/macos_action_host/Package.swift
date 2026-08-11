// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "JarvisActionNative",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "jarvis-action-native", targets: ["JarvisActionNative"])],
    targets: [.executableTarget(name: "JarvisActionNative")]
)
