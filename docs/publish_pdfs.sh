#!/bin/bash
# Copy the latest compiled PDFs into docs/papers and push them if they changed.
cd "$(dirname "$0")/.."
cp paper_A/paper_A.pdf docs/papers/paper_A.pdf 2>/dev/null
cp paper_B/paper_B.pdf docs/papers/paper_B.pdf 2>/dev/null
git add docs/papers/*.pdf
if ! git diff --cached --quiet -- docs/papers; then
  git commit -q -m "Update draft PDFs ($(date '+%Y-%m-%d %H:%M'))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- docs/papers && git push -q && echo "pushed $(date +%H:%M)"
else
  echo "unchanged $(date +%H:%M)"
fi
