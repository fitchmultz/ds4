#!/bin/sh
set -e

# GLM-5.2 GGUF downloader (Unsloth UD-IQ2_M, glm-dsa arch).
# Repo is public + ungated: https://huggingface.co/unsloth/GLM-5.2-GGUF
# Uses curl with HTTP/1.1 + resume (-C -) so no hf/huggingface-cli dependency.

REPO="unsloth/GLM-5.2-GGUF"
QUANT="${GLM52_QUANT:-UD-IQ2_M}"
SHARDS=6
TOTAL_BYTES=238577580768   # UD-IQ2_M full split set, ~222 GiB

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${GLM52_GGUF_DIR:-"$ROOT/gguf/glm52"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
SHARD1_ONLY=0
TOKEN=${HF_TOKEN:-${HUGGING_FACE_TOKEN:-}}

usage() {
    cat <<EOF
GLM-5.2 GGUF downloader (glm-dsa, $REPO)

Usage:
  ./download_glm52.sh [--shard1-only] [--token TOKEN]

Targets:
  $QUANT   2-bit dynamic quant, 6 shards, ~222 GiB / 238 GB total.
           The guide's ~239GB / ~82% accuracy target. Runs on a 256GB Mac or
           RAM/VRAM setups; on a 128GB Mac you must use --ssd-streaming.

Options:
  --shard1-only   Fetch only shard 1 (metadata+tokenizer, ~9 MB). Used to
                  validate the loader / --inspect without the full model.
  --token TOKEN   Bearer token. Unnecessary: the repo is public and ungated.
  --quant NAME    Quant subdir (default UD-IQ2_M).

Environment:
  GLM52_GGUF_DIR  Output dir (default ./gguf/glm52)
  HF_TOKEN / HUGGING_FACE_TOKEN   Bearer token if you want to send one.

After download, the first shard is the loader entry:
  ./ds4 --inspect -m $OUT_DIR/GLM-5.2-${QUANT}-00001-of-00006.gguf
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --shard1-only) SHARD1_ONLY=1 ;;
        --token) shift; TOKEN=$1 ;;
        --quant) shift; QUANT=$1 ;;
        -h|--help|help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUT_DIR"

shard_name() { printf 'GLM-5.2-%s-%05d-of-%05d.gguf' "$QUANT" "$1" "$SHARDS"; }

auth_header() {
    [ -n "$TOKEN" ] && printf ' -H "Authorization: Bearer %s"' "$TOKEN"
}

download_shard() {
    idx=$1
    file=$(shard_name "$idx")
    out="$OUT_DIR/$file"
    part="$out.part"
    url="https://huggingface.co/$REPO/resolve/main/$QUANT/$file"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from $url (curl, HTTP/1.1, resume with -C -)"

    # shellcheck disable=SC2086
    if [ -n "$TOKEN" ]; then
        curl -fL --http1.1 -C - --progress-bar \
            -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --http1.1 -C - --progress-bar -o "$part" "$url"
    fi

    mv "$part" "$out"
}

if [ "$SHARD1_ONLY" -eq 1 ]; then
    echo "Shard-1 only: metadata + tokenizer (~9 MB)."
    download_shard 1
    echo
    echo "Done. Loader entry: $OUT_DIR/$(shard_name 1)"
    exit 0
fi

printf 'Full %s split: %d shards, %.1f GiB total. Resumable.\n' \
    "$QUANT" "$SHARDS" "$(echo "scale=1; $TOTAL_BYTES/1073741824" | bc)"

idx=1
while [ "$idx" -le "$SHARDS" ]; do
    download_shard "$idx"
    idx=$((idx + 1))
done

echo
echo "Done. Loader entry: $OUT_DIR/$(shard_name 1)"
echo "Run on 128GB Mac with: ./ds4 --ssd-streaming -m $OUT_DIR/$(shard_name 1) -p \"Hello\""
