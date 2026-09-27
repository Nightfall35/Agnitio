#!/usr/bin/env bash
# Quick manual validation of the facial recognition service.
#
# Usage:
#   ./test_pipeline.sh enroll "Jane Doe" path/to/jane1.jpg
#   ./test_pipeline.sh search path/to/jane_query.jpg [top_k]
#
# Requires: curl, jq
# Override the service URL with FACE_API_URL (default http://localhost:8000)

set -euo pipefail

API_URL="${FACE_API_URL:-http://localhost:8000}"

usage() {
    echo "Usage:"
    echo "  $0 enroll <name> <image_path>"
    echo "  $0 search <image_path> [top_k]"
    exit 1
}

[ $# -lt 2 ] && usage

command="$1"

case "$command" in
    enroll)
        name="$2"
        image_path="${3:-}"
        [ -z "$image_path" ] && usage
        echo "--- ENROLL: $name <- $image_path ---"
        curl -s -X POST "$API_URL/enroll" \
            -F "name=$name" \
            -F "file=@$image_path" | jq .
        ;;

    search)
        image_path="$2"
        top_k="${3:-5}"
        echo "--- SEARCH: $image_path (top_k=$top_k) ---"
        response=$(curl -s -X POST "$API_URL/search" \
            -F "top_k=$top_k" \
            -F "file=@$image_path")
        echo "$response" | jq .

        threshold=$(echo "$response" | jq -r '.threshold')
        echo ""
        echo "Threshold: $threshold"
        echo "$response" | jq -r '.results[] | "\(.name)\tsimilarity=\(.similarity)\tis_match=\(.is_match)"'
        ;;

    *)
        usage
        ;;
esac
