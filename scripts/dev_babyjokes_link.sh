#!/usr/bin/env bash
# The BabyJokes project and its runs read the clips from /root/data/babyjokes/public,
# which a container rebuild removes; the clips themselves live in temp/ in the
# workspace. Link them again if they're there. Never fails the container's start.
clips="$(cd "$(dirname "$0")/.." && pwd)/temp/babyjokes/public"
if [ -d "$clips" ]; then
  mkdir -p /root/data/babyjokes 2>/dev/null && ln -sfn "$clips" /root/data/babyjokes/public 2>/dev/null \
    && echo "BabyJokes clips linked at /root/data/babyjokes/public"
fi
exit 0
