# SARC — smart secure archiver (Project 2)

| folder | content |
|---|---|
| `python/` | tested reference implementation (28 tests pass), GA, attacks, report generator — see `python/README.md` |
| `csharp/` | .NET 9 port of the core (NOT compiled yet — run `selftest` first) — see `csharp/README.md` |
| `corpus/ train/ valid/ reference/ team_config.json` | teacher-issued data and reference archives |

Quick start (Python): `cd python && python3 tests/test_sarc.py`
Quick start (C#): `cd csharp && dotnet run -c Release -- selftest ..`
`python/META-USAGE.md` documents the AI assistance used.
