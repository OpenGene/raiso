#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -s)" == Darwin ]] || { echo 'This script requires macOS' >&2; exit 1; }
: "${CONDA_PREFIX:?Activate the environment from environment-macos.yml first}"
cd "$(dirname "$0")/.."

case "$(uname -m)" in
    arm64) target=aarch64-apple-darwin ;;
    x86_64) target=x86_64-apple-darwin ;;
    *) echo 'Unsupported macOS architecture' >&2; exit 1 ;;
esac

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
export HDF5_DIR="${CONDA_PREFIX}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export CC="${CC:-$(xcrun --find clang)}"
export CXX="${CXX:-$(xcrun --find clang++)}"
export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"
if [[ "$target" == aarch64-apple-darwin ]]; then
    export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$CC"
else
    export CARGO_TARGET_X86_64_APPLE_DARWIN_LINKER="$CC"
fi

cargo build --release --locked --bin riaso --target "$target"
out="$PWD/dist-macos/$target"
mkdir -p "$out/bin" "$out/lib" "$out/fonts" "$out/etc/fonts"
cp "$CARGO_TARGET_DIR/$target/release/riaso" "$out/bin/riaso"
chmod 755 "$out/bin/riaso"
delocate-path --executable-path "$out/bin" -L lib "$out"

python - "$CONDA_PREFIX" "$out" <<'PY'
from pathlib import Path
import shutil
import sys

prefix, out = map(Path, sys.argv[1:])
fonts = sorted((prefix / "fonts").glob("*.ttf"))
if not fonts:
    fonts = sorted((prefix / "share/fonts").rglob("*.ttf"))
if not fonts:
    raise SystemExit("No packaged fonts found")
for font in fonts:
    shutil.copy2(font, out / "fonts" / font.name)
(out / "etc/fonts/fonts.conf").write_text(
    '<?xml version="1.0"?>\n<fontconfig><dir prefix="relative">../../fonts</dir></fontconfig>\n',
    encoding="utf-8",
)
for license in (Path("LICENSE"), Path("../LICENSE"), Path("../../LICENSE")):
    if license.is_file():
        shutil.copy2(license, out / "LICENSE")
        break
PY

cat > "$out/riaso" <<'SH'
#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
export FONTCONFIG_FILE="$here/etc/fonts/fonts.conf"
exec "$here/bin/riaso" "$@"
SH
chmod 755 "$out/riaso"
env -i HOME="$HOME" "$out/riaso" --version
env -i HOME="$HOME" "$out/riaso" --HELP > /dev/null
printf 'Distribution: %s\n' "$out"
