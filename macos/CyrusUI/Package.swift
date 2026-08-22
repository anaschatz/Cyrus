// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "CyrusUI",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "CyrusUI", targets: ["CyrusUI"]),
    ],
    targets: [
        .executableTarget(name: "CyrusUI"),
        .testTarget(name: "CyrusUITests", dependencies: ["CyrusUI"]),
    ]
)
