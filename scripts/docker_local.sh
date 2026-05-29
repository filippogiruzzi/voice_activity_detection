#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

docker run --rm \
    -v "$ROOT":/app \
    -it \
    --entrypoint /bin/bash \
    vad
