#!/usr/bin/env bash
# spread_dates.sh
#
# Rewrites author/committer dates of commits on the current branch so they
# appear spread evenly across 2024, newest commit last.
#
# WARNING: this rewrites history. After running you must force-push:
#   git push --force --force-with-lease
# Anyone who already cloned the repo will see a divergent history.
#
# Usage:
#   bash spread_dates.sh            # dry-run: print planned dates
#   bash spread_dates.sh --apply    # actually rewrite

set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

# Oldest -> newest
mapfile -t COMMITS < <(git rev-list --reverse HEAD)
N=${#COMMITS[@]}

if (( N == 0 )); then
  echo "No commits found."
  exit 0
fi

# Spread from 2024-01-15 10:00 to 2024-12-20 18:00
START=$(date -u -d "2024-01-15 10:00:00" +%s 2>/dev/null || date -u -j -f "%Y-%m-%d %H:%M:%S" "2024-01-15 10:00:00" +%s)
END=$(date -u -d "2024-12-20 18:00:00" +%s 2>/dev/null || date -u -j -f "%Y-%m-%d %H:%M:%S" "2024-12-20 18:00:00" +%s)
SPAN=$(( END - START ))

echo "Rewriting $N commits across 2024..."

i=0
for c in "${COMMITS[@]}"; do
  # Evenly spaced, with a small random jitter of +/- 6 hours
  base=$(( START + SPAN * i / (N > 1 ? N - 1 : 1) ))
  jitter=$(( (RANDOM % 43200) - 21600 ))   # seconds
  ts=$(( base + jitter ))
  # Keep it inside 2024
  if (( ts < START )); then ts=$START; fi
  if (( ts > END )); then ts=$END; fi

  date_str=$(date -u -d "@$ts" "+%Y-%m-%dT%H:%M:%S" 2>/dev/null || date -u -r "$ts" "+%Y-%m-%dT%H:%M:%S")

  if (( APPLY == 0 )); then
    echo "  $c  ->  $date_str"
  else
    FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
      --env-filter "
        if [ \"\$GIT_COMMIT\" = \"$c\" ]; then
          export GIT_AUTHOR_DATE=\"$date_str\"
          export GIT_COMMITTER_DATE=\"$date_str\"
        fi
      " -- --all >/dev/null
    echo "  $c  ->  $date_str  (done)"
  fi
  i=$(( i + 1 ))
done

if (( APPLY == 1 )); then
  echo
  echo "Done. Verify with: git log --format='%h %ad %s' --date=short"
  echo "Then force-push: git push --force --force-with-lease"
else
  echo
  echo "Dry run only. Re-run with --apply to rewrite."
fi
