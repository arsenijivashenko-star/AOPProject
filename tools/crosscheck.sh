#!/usr/bin/env bash
# Python <-> C# cross-compatibility: an archive made by one implementation must open in the other, byte-exact.
# Usage (from the repo root):  tools/crosscheck.sh        needs: python3 (+cryptography, numpy) and the .NET 9 SDK.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PW=algo2026
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
CS=(dotnet run --project "$ROOT/csharp" -c Release --no-build --)
PY=(python3 "$ROOT/python/sarc_cli.py")

dotnet build "$ROOT/csharp" -c Release -nologo -v quiet

for strat in lazy dp-iter; do
  echo "== Python ($strat) -> C#"
  "${PY[@]}" archive "$ROOT/corpus" "$W/py_$strat" -p $PW --strategy "$strat" >/dev/null
  "${CS[@]}" extract "$W/py_$strat" "$W/cs_out_$strat" -p $PW
  diff -r "$ROOT/corpus" "$W/cs_out_$strat" && echo "identical"

  echo "== C# ($strat) -> Python"
  "${CS[@]}" archive "$ROOT/corpus" "$W/cs_$strat" -p $PW --strategy "$strat" >/dev/null
  "${PY[@]}" extract "$W/cs_$strat" "$W/py_out_$strat" -p $PW
  diff -r "$ROOT/corpus" "$W/py_out_$strat" && echo "identical"

  echo "== list/verify on both sides ($strat)"
  "${PY[@]}" verify "$W/cs_$strat" -p $PW; "${CS[@]}" verify "$W/py_$strat" -p $PW
done
echo "cross-check OK"
