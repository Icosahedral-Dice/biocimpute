#!/usr/bin/env bash
# Install the reference R package impute into a local (gitignored) library,
# used only to (re)generate the parity fixtures in tests/fixtures.
set -euo pipefail
VERSION="${IMPUTE_VERSION:-1.86.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LIB="$HERE/lib"
mkdir -p "$LIB"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -sSfL -o "$TMP/impute.tar.gz" \
  "https://bioconductor.org/packages/release/bioc/src/contrib/impute_${VERSION}.tar.gz"
R CMD INSTALL -l "$LIB" "$TMP/impute.tar.gz"
