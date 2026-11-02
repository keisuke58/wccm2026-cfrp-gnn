#!/usr/bin/env bash
# fetch_papers.sh -- download the papers summarised in research/DENSO_PUBLIC_INFO.md (free on
# J-STAGE). The PDFs are NOT committed: the repository is public and the papers are copyrighted.
#   bash research/papers/fetch_papers.sh        # -> research/papers/*.pdf (gitignored)
set -euo pipefail
cd "$(dirname "$0")"
get() { curl -fsSL -o "$1" "$2" && echo "downloaded $1"; }
# 津田ほか（デンソー・ポリプラスチックス）, 日本複合材料学会誌 52(3), 88-96, 2026
get tsuda2026_jscm52_88.pdf "https://www.jstage.jst.go.jp/article/jscm/52/3/52_88/_pdf/-char/ja"
# 田中ほか（名城大学・横浜国立大学）, 材料 63(7), 514-520, 2014
get tanaka2014_jsms63_514.pdf "https://www.jstage.jst.go.jp/article/jsms/63/7/63_514/_pdf/-char/ja"
