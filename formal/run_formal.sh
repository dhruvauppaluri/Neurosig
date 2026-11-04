#!/usr/bin/env bash
# Prove the SVA contracts in this directory.
# Requires Yosys with the slang plugin, SymbiYosys, and bitwuzla.
# The OSS CAD Suite provides that set. Point OSS_CAD_SUITE at its top directory
# if those tools are not already on PATH.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
out=${FORMAL_WORKDIR:-/tmp/neurosig-formal}

if [[ -n "${OSS_CAD_SUITE:-}" ]]; then
    # shellcheck disable=SC1091
    source "$OSS_CAD_SUITE/environment"
fi

if ! command -v sby >/dev/null || ! command -v yosys >/dev/null; then
    echo "sby and yosys are required. Install the OSS CAD Suite and set OSS_CAD_SUITE." >&2
    exit 1
fi

mkdir -p "$out"
cd "$root"
# .sby [files] paths are relative to this directory.

sby -f --prefix "$out/spike_count_decoder" formal/spike_count_decoder.sby
sby -f --prefix "$out/biquad_controller" formal/biquad_controller.sby

echo "Formal proofs passed. Logs: $out"
