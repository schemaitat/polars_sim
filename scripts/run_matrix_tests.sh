#!/usr/bin/env bash
# Run polars version matrix tests
set -euo pipefail

n="${1:-all}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname "$script_dir")"

# Get all polars versions >= 1.0.0
echo "Fetching polars versions..."
all_versions=$(uvx pip index versions polars --json | \
    python3 "$script_dir/get_polars_versions.py")

if [ -z "$all_versions" ]; then
    echo "Error: Could not fetch polars versions"
    exit 1
fi

# Filter to top N versions if specified
if [ "$n" != "all" ]; then
    versions=$(echo "$all_versions" | head -n "$n")
    echo "Testing top $n versions"
else
    versions="$all_versions"
    version_count=$(echo "$versions" | wc -l | tr -d ' ')
    echo "Testing all $version_count versions"
fi

# Create temp directory for results
tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Get the path to just command
just_cmd=$(which just)

# Start tests in parallel
for version in $versions; do
    printf "Testing polars version %s...\n" "$version"
    (
        start=$(date +%s)
        if cd "$project_dir" && "$just_cmd" test-polars-version "$version" \
            > "$tmpdir/$version.log" 2>&1; then
            end=$(date +%s)
            duration=$((end - start))
            echo "SUCCESS $duration" > "$tmpdir/$version.status"
        else
            end=$(date +%s)
            duration=$((end - start))
            echo "FAILURE $duration" > "$tmpdir/$version.status"
        fi
    ) &
done

# Wait for all tests
wait

# Print results using Python script
python3 "$script_dir/print_test_results.py" "$tmpdir" $versions
