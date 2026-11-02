#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -s)" == Darwin ]] || { echo 'This script requires macOS' >&2; exit 1; }
: "${CONDA_PREFIX:?Activate the environment from environment-macos.yml first}"
cd "$(dirname "$0")/.."

case "$(uname -m)" in
    arm64) target=aarch64-apple-darwin; arch=arm64 ;;
    x86_64) target=x86_64-apple-darwin; arch=x86_64 ;;
    *) echo 'Unsupported macOS architecture' >&2; exit 1 ;;
esac

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
export HDF5_DIR="${CONDA_PREFIX}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CC="${CC:-$(xcrun --find clang)}"
export CXX="${CXX:-$(xcrun --find clang++)}"
export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"
export PYO3_PYTHON="$(command -v python)"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
if [[ "$arch" == arm64 ]]; then
    export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$CC"
else
    export CARGO_TARGET_X86_64_APPLE_DARWIN_LINKER="$CC"
fi

python - <<'PY'
import json
import os
from pathlib import Path
import subprocess

prefix = Path(os.environ["CONDA_PREFIX"])
for name in ("libblas", "libcblas", "liblapack", "libopenblas"):
    records = list((prefix / "conda-meta").glob(f"{name}-*.json"))
    if len(records) != 1:
        raise SystemExit(f"Expected one installed {name} record: {records}")
    record = json.loads(records[0].read_text())
    print(f"{name}: {record['version']} {record['build']}", flush=True)
    if not (record["build"].startswith("pthreads") if name == "libopenblas" else record["build"].endswith("_openblas")):
        raise SystemExit(f"{name} must use pthread OpenBLAS; recreate/update environment-macos.yml")
for name in ("libblas.3.dylib", "liblapack.3.dylib", "libarpack.2.dylib"):
    library = prefix / "lib" / name
    print(f"{library} -> {library.resolve()}", flush=True)
    subprocess.run(["otool", "-L", str(library)], check=True)
PY

python scripts/prepare_share.py
trap 'python scripts/prepare_share.py --clean' EXIT
raw=$(mktemp -d "${TMPDIR:-/tmp}/riasopy-wheel.XXXXXX")
trap 'python scripts/prepare_share.py --clean; rm -rf "$raw"' EXIT
maturin build --release --locked --target "$target" --interpreter "$PYO3_PYTHON" --out "$raw"
mkdir -p target/wheels-macos
# Let delocate resolve @rpath dependencies from the build environment.
DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" delocate-wheel --require-archs "$arch" --executable-path "$CONDA_PREFIX/bin" \
    -w target/wheels-macos "$raw"/riasopy-*.whl
python - <<'PY'
from pathlib import Path
import zipfile
import tempfile
from delocate.delocating import _get_macos_min_version

for wheel in Path("target/wheels-macos").glob("riasopy-*.whl"):
    with zipfile.ZipFile(wheel) as archive:
        runtimes = [name for name in archive.namelist()
                    if name.endswith(".dylib") and
                    any(token in Path(name).name.lower()
                        for token in ("libomp", "libiomp", "libgomp"))]
    if runtimes:
        raise SystemExit(f"Unexpected bundled OpenMP runtime in {wheel}: {runtimes}")
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(temporary)
        for library in Path(temporary).rglob("*"):
            if library.suffix not in (".so", ".dylib"):
                continue
            versions = list(_get_macos_min_version(library))
            if not versions:
                raise SystemExit(f"Cannot determine minimum macOS version: {library.name}")
            for architecture, version in versions:
                if version.release > (11, 0):
                    raise SystemExit(f"{library.name} ({architecture}) requires macOS {version}, expected <=11.0")
    print(f"Verified no bundled OpenMP runtime: {wheel.name}")
PY
printf 'Wheels: %s/target/wheels-macos\n' "$PWD"
