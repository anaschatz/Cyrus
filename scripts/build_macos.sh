#!/bin/zsh
set -euo pipefail

repo_root=${0:A:h:h}
cd "$repo_root"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "Native macOS arm64 is required."
  exit 2
fi

: "${JARVIS_MACOS_WHISPER_SERVER:?Set an absolute path to the pinned native whisper-server.}"
: "${JARVIS_MACOS_CODESIGN_IDENTITY:?Set the Developer ID Application identity.}"
: "${JARVIS_MACOS_NOTARY_PROFILE:?Set the notarytool keychain profile.}"

python_version=$(python3.11 -c 'import platform; print(platform.python_version())')
architecture=$(python3.11 -c 'import platform; print(platform.machine())')
pyinstaller_version=$(python3.11 -c 'import PyInstaller; print(PyInstaller.__version__)')
if [[ "$python_version" != "3.11.14" || "$architecture" != "arm64" || "$pyinstaller_version" != "6.21.0" ]]; then
  print -u2 "Expected CPython 3.11.14 arm64 and PyInstaller 6.21.0. Nothing was installed."
  exit 2
fi
if [[ "$JARVIS_MACOS_WHISPER_SERVER" != /* || ! -f "$JARVIS_MACOS_WHISPER_SERVER" || -L "$JARVIS_MACOS_WHISPER_SERVER" ]]; then
  print -u2 "whisper-server must be an absolute regular non-symlink file."
  exit 2
fi
if ! /usr/bin/file "$JARVIS_MACOS_WHISPER_SERVER" | /usr/bin/grep -q 'Mach-O.*arm64'; then
  print -u2 "whisper-server is not a native arm64 Mach-O helper."
  exit 2
fi

build_root=$(mktemp -d "${TMPDIR:-/tmp}/jarvis-macos-build.XXXXXX")
trap 'rm -rf "$build_root"' EXIT
export JARVIS_MACOS_BUILD_PROFILE="$build_root/jarvis-macos-runtime-profile.json"
export JARVIS_MACOS_ACTION_HELPER_MANIFEST="$build_root/jarvis-action-helper.json"
export PYTHONPATH="$repo_root/src:$repo_root/orchestrator/src"

/usr/bin/swift build \
  --package-path "$repo_root/native/macos_action_host" \
  --scratch-path "$build_root/swift" \
  --configuration release
export JARVIS_MACOS_ACTION_HELPER="$build_root/swift/arm64-apple-macosx/release/jarvis-action-native"
if [[ ! -x "$JARVIS_MACOS_ACTION_HELPER" || -L "$JARVIS_MACOS_ACTION_HELPER" ]]; then
  print -u2 "The pinned Swift helper build did not produce the expected executable."
  exit 2
fi
/usr/bin/codesign --force --options runtime --timestamp \
  --sign "$JARVIS_MACOS_CODESIGN_IDENTITY" \
  --entitlements "$repo_root/native/macos_action_host/JarvisActionNative.entitlements" \
  "$JARVIS_MACOS_ACTION_HELPER"
python3.11 scripts/inspect_macos_helper.py \
  --helper "$JARVIS_MACOS_ACTION_HELPER" \
  --output "$JARVIS_MACOS_ACTION_HELPER_MANIFEST"

python3.11 scripts/build_macos_profile.py \
  --helper "$JARVIS_MACOS_WHISPER_SERVER" \
  --pyinstaller-version "$pyinstaller_version" \
  --output "$JARVIS_MACOS_BUILD_PROFILE"
python3.11 -m unittest discover -s tests -v
python3.11 -m compileall -q src tests packaging
python3.11 -m PyInstaller --clean --noconfirm packaging/jarvis_voice_macos.spec

app="$repo_root/dist/JARVIS Voice.app"
embedded_action_helper="$app/Contents/Frameworks/helpers/jarvis-action-native"
python3.11 scripts/inspect_macos_helper.py \
  --helper "$embedded_action_helper" \
  --output "$repo_root/dist/jarvis-action-helper-artifact.json"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$app"
/usr/sbin/spctl --assess --type execute --verbose=2 "$app"
/usr/bin/ditto -c -k --keepParent "$app" "$build_root/JARVIS-Voice.zip"
/usr/bin/xcrun notarytool submit "$build_root/JARVIS-Voice.zip" \
  --keychain-profile "$JARVIS_MACOS_NOTARY_PROFILE" --wait
/usr/bin/xcrun stapler staple "$app"
/usr/bin/xcrun stapler validate "$app"

print "Signed and notarized app created at: $app"
print "No model, user config, raw audio, package installation, or network model acquisition was included."
