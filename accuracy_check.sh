#!/usr/bin/env bash
# Batch accuracy check: enrolls a labeled dataset, then searches every
# photo back against the database and reports correct/wrong/no-match counts.
#
# Expected layout:
#   dataset/
#       Jane Doe/
#           1.jpg
#           2.jpg
#       John Smith/
#           1.jpg
#
# Usage:
#   ./accuracy_check.sh /path/to/dataset
#
# Requires: curl, jq
# Override the service URL with FACE_API_URL (default http://localhost:8000)

set -euo pipefail

API_URL="${FACE_API_URL:-http://localhost:8000}"
DATASET_DIR="${1:-}"

if [ -z "$DATASET_DIR" ] || [ ! -d "$DATASET_DIR" ]; then
    echo "Usage: $0 /path/to/dataset"
    exit 1
fi

enrolled_paths=()
enrolled_names=()

echo "Enrolling dataset..."
for person_dir in "$DATASET_DIR"/*/; do
    person_name=$(basename "$person_dir")
    for image_path in "$person_dir"*; do
        [ -f "$image_path" ] || continue
        resp=$(curl -s -X POST "$API_URL/enroll" -F "name=$person_name" -F "file=@$image_path")
        if echo "$resp" | jq -e '.person_id' > /dev/null 2>&1; then
            echo "[enrolled] $person_name <- $(basename "$image_path")"
            enrolled_paths+=("$image_path")
            enrolled_names+=("$person_name")
        else
            echo "[FAILED to enroll] $person_name <- $(basename "$image_path"): $resp"
        fi
    done
done

echo ""
echo "Running accuracy check..."
echo ""

correct=0
incorrect=0
no_match=0

for i in "${!enrolled_paths[@]}"; do
    image_path="${enrolled_paths[$i]}"
    true_name="${enrolled_names[$i]}"

    resp=$(curl -s -X POST "$API_URL/search" -F "top_k=1" -F "file=@$image_path")
    top_name=$(echo "$resp" | jq -r '.results[0].name // empty')
    is_match=$(echo "$resp" | jq -r '.results[0].is_match // false')
    similarity=$(echo "$resp" | jq -r '.results[0].similarity // "n/a"')

    if [ -z "$top_name" ] || [ "$is_match" != "true" ]; then
        echo "[NO MATCH] $true_name <- $(basename "$image_path")"
        no_match=$((no_match + 1))
    elif [ "$top_name" == "$true_name" ]; then
        echo "[correct]  $true_name <- $(basename "$image_path")  (similarity=$similarity)"
        correct=$((correct + 1))
    else
        echo "[WRONG]    $true_name <- $(basename "$image_path")  matched to $top_name (similarity=$similarity)"
        incorrect=$((incorrect + 1))
    fi
done

total=$((correct + incorrect + no_match))
echo ""
echo "--- Summary ---"
echo "Total tested:    $total"
echo "Correct matches: $correct"
echo "Wrong matches:   $incorrect"
echo "No match found:  $no_match"
if [ "$total" -gt 0 ]; then
    awk -v c="$correct" -v t="$total" 'BEGIN { printf "Accuracy:        %.1f%%\n", (c/t)*100 }'
fi
