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

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
export HDF5_DIR="${CONDA_PREFIX}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CC="${CC:-$(xcrun --find clang)}"
export CXX="${CXX:-$(xcrun --find clang++)}"
export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"
export PYO3_PYTHON="$(command -v python)"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
# Reserve extra Mach-O load-command space so delocate can rewrite the
# dependent-library install names when bundling the wheel.
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-headerpad_max_install_names"
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
for name in ("libblas", "libcblas", "liblapack"):
    records = list((prefix / "conda-meta").glob(f"{name}-*.json"))
    if len(records) != 1:
        raise SystemExit(f"Expected one installed {name} record: {records}")
    record = json.loads(records[0].read_text())
    print(f"{name}: {record['version']} {record['build']}", flush=True)
    if not record["build"].endswith("_openblas"):
        raise SystemExit(f"{name} must use the OpenBLAS variant; recreate/update environment-macos.yml")
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
# Conda's BLAS package exposes libblas/liblapack as symlink aliases to the
# OpenBLAS binary.  delocate may copy only the real SONAME (libopenblas.0),
# while libarpack still refers to the aliases.  Add those aliases to the wheel
# so the loader can resolve the dependency on a clean machine.
python - <<'PY'
from pathlib import Path
import tempfile
import zipfile
import base64
import hashlib

for wheel in Path("target/wheels-macos").glob("riasopy-*.whl"):
    with zipfile.ZipFile(wheel, "r") as src:
        names = set(src.namelist())
        dylibs = [n for n in names if n.startswith("riasopy/.dylibs/") and
                  Path(n).name.startswith("libopenblas") and n.endswith(".dylib")]
        if not dylibs:
            raise SystemExit(f"No bundled OpenBLAS library found in {wheel}")
        source = max(dylibs, key=lambda n: len(src.read(n)))
        payload = src.read(source)
        additions = {}
        for alias in ("libblas.3.dylib", "liblapack.3.dylib", "libcblas.3.dylib"):
            name = f"riasopy/.dylibs/{alias}"
            if name not in names:
                additions[name] = payload
        if not additions:
            continue
        with tempfile.NamedTemporaryFile(dir=wheel.parent, suffix=".whl", delete=False) as tmp:
            replacement = Path(tmp.name)
        with zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                if info.filename.endswith(".dist-info/RECORD"):
                    continue
                dst.writestr(info, src.read(info.filename))
            for name, data in additions.items():
                dst.writestr(name, data)
            record_name = next(n for n in names if n.endswith(".dist-info/RECORD"))
            records = src.read(record_name).decode().splitlines()
            for name, data in additions.items():
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                records.append(f"{name},sha256={digest},{len(data)}")
            records.append(f"{record_name},,")
            dst.writestr(record_name, "\n".join(records) + "\n")
        replacement.replace(wheel)
        print(f"Added BLAS aliases to {wheel.name}: {', '.join(additions)}")
PY
python - <<'PY'
from pathlib import Path
import zipfile

for wheel in Path("target/wheels-macos").glob("riasopy-*.whl"):
    with zipfile.ZipFile(wheel) as archive:
        runtimes = [name for name in archive.namelist()
                    if name.endswith(".dylib") and
                    any(token in Path(name).name.lower()
                        for token in ("libomp", "libiomp", "libgomp"))]
    if runtimes:
        raise SystemExit(f"Unexpected bundled OpenMP runtime in {wheel}: {runtimes}")
    print(f"Verified no bundled OpenMP runtime: {wheel.name}")
PY
printf 'Wheels: %s/target/wheels-macos\n' "$PWD"
