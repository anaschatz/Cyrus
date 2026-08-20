// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "JarvisUI",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "JarvisUI", targets: ["JarvisUI"]),
    ],
    targets: [
        .executableTarget(name: "JarvisUI"),
        .testTarget(name: "JarvisUITests", dependencies: ["JarvisUI"]),
    ]
)
