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
# Nohara et al., Polimeros 16(2), 104-110, 2006 (SciELO, free)
get nohara2006_polimeros16_104.pdf "https://www.scielo.br/j/po/a/kzSJKJHTpkBdzHvDrVH875M/?format=pdf&lang=en"
# Elsevier (subscription; fetch by hand from a university network, keep in research/papers/, gitignored):
#   lopez1988_polymer29_106.pdf  doi:10.1016/0032-3861(88)90207-8
#   lopez1989_polymer30_147.pdf  doi:10.1016/0032-3861(89)90395-9
