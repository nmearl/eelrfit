#!/usr/bin/env bash
# Download, build and smoke-test the Cloudy photoionization code.
#
# Usage: scripts/install_cloudy.sh [-v VERSION] [-p PREFIX] [-j JOBS] [-a ARCHIVE]
#
#   -v  Cloudy release (default c25.00)
#   -p  install directory (default ~/.local/share/cloudy); Cloudy ends up in PREFIX/VERSION
#   -j  parallel make jobs (default: all CPUs)
#   -a  use an already downloaded release archive (e.g. on cluster nodes without internet)
#
# Needs curl, make, perl and a C++17 compiler. Cloudy compiles its own location
# into the executable, so do not move the installation afterwards. Its Makefile
# may build for the local CPU (-march=native): build on the kind of machine
# that will run the grids. A build takes 10-30 minutes and about 4 GB.
set -euo pipefail

version="c25.00"
prefix="$HOME/.local/share/cloudy"
jobs=""
archive=""
while getopts "v:p:j:a:h" opt; do
    case "$opt" in
        v) version="$OPTARG" ;;
        p) prefix="$OPTARG" ;;
        j) jobs="$OPTARG" ;;
        a) archive="$OPTARG" ;;
        *) sed -n '2,10p' "$0"; exit 1 ;;
    esac
done

# SHA-256 of the official archives, computed from downloads (Cloudy does not publish checksums).
case "$version" in
    c25.00) checksum="12a4fac7a29f888f56b37885df0067c92eae47a53c78060743e3c92140add5f3" ;;
    c23.01) checksum="a9ad2dc037e88f552389de0e483d68f54310976dee32154ed13830882aedae0b" ;;
    *) checksum=""; echo "Note: no known checksum for $version; the archive will not be verified." ;;
esac

for tool in make perl curl; do
    command -v "$tool" >/dev/null || { echo "$tool is required"; exit 1; }
done
command -v c++ >/dev/null || command -v g++ >/dev/null || command -v clang++ >/dev/null || { echo "A C++ compiler is required"; exit 1; }
case "$prefix" in *[[:space:]]*) echo "Cloudy cannot be built in a path containing spaces: $prefix"; exit 1 ;; esac

root="$prefix/$version"
if [[ -x "$root/source/cloudy.exe" ]]; then
    echo "Cloudy $version is already built in $root"
    echo "export CLOUDY_EXE=$root/source/cloudy.exe"
    exit 0
fi
mkdir -p "$prefix"

if [[ -z "$archive" ]]; then
    archive="$prefix/$version.tar.gz"
    major="${version%%.*}"
    echo "Downloading Cloudy $version"
    curl -fL --progress-bar -o "$archive" "https://data.nublado.org/cloudy_releases/$major/$version.tar.gz"
fi
if [[ -n "$checksum" ]]; then
    actual=$( (command -v sha256sum >/dev/null && sha256sum "$archive" || shasum -a 256 "$archive") | cut -d' ' -f1)
    [[ "$actual" == "$checksum" ]] || { echo "Checksum mismatch for $archive: $actual"; exit 1; }
fi

echo "Extracting to $root"
tar -xzf "$archive" -C "$prefix"
[[ -d "$root/source" ]] || { echo "The archive did not contain $version/source"; exit 1; }

jobs="${jobs:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
echo "Building with $jobs jobs (log: $root/build.log)"
(cd "$root/source" && make -j "$jobs") > "$root/build.log" 2>&1 || { tail -20 "$root/build.log"; echo "Build failed; see $root/build.log"; exit 1; }
grep -q -- "-march=native" "$root/build.log" && echo "Note: built with -march=native; run it on this kind of CPU."

echo "Running Cloudy's smoke test"
workdir=$(mktemp -d)
(cd "$workdir" && printf 'test\n\n' | "$root/source/cloudy.exe") > "$root/smoke_test.out" 2>&1 || true
rm -rf "$workdir"
grep -q "Cloudy exited OK" "$root/smoke_test.out" || { echo "The smoke test failed; see $root/smoke_test.out"; exit 1; }

echo "Cloudy $version is ready. Add this to your shell profile:"
echo "export CLOUDY_EXE=$root/source/cloudy.exe"
