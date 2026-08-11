// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "JarvisSchedulerNative",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "jarvis-scheduler-native", targets: ["JarvisSchedulerNative"])],
    targets: [.executableTarget(name: "JarvisSchedulerNative")]
)
