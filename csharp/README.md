# SarcArchiver — C# (.NET 9) port

**Status: written and statically reviewed, but NOT yet compiled or run** (the authoring environment has no .NET SDK and no network).
A line-by-line review fixed one likely compile error (tuple-name mismatch in `ParseLazy`) and checked feature-vector parity with
the Python classifier. `.github/workflows/ci.yml` builds and self-tests it on every push; `tools/crosscheck.sh` checks Python<->C#
archive compatibility. Expect to fix a few compile errors on the first `dotnet build` if CI is not green yet. The logic is a line-by-line port of the Python version, which *was* tested
(28 tests, all reference archives). Please run `selftest` first; it will tell you quickly what works.

## Build / run
```
dotnet build -c Release
dotnet run -c Release -- selftest /path/to/Project        # Project/ = folder with corpus/ and reference/
dotnet run -c Release -- archive /path/to/corpus out/c -p algo2026
dotnet run -c Release -- extract out/c out/restored -p algo2026
dotnet run -c Release -- list out/c -p algo2026        # file table
dotnet run -c Release -- verify out/c -p algo2026      # full integrity check, writes nothing
dotnet run -c Release -- archive /path/to/corpus out/c -p algo2026 --strategy dp-iter   # densest parser
dotnet run -c Release -- info out/c.part0.arc
dotnet run -c Release -- attack integrity out/c.part0.arc -p algo2026
```
No NuGet packages are used (AES, HMAC, PBKDF2, JSON all come from the .NET base class library), so it builds offline.
`model.json` (perceptron weights) is copied next to the executable by the .csproj.

## Layout (src/)
| file | content |
|---|---|
| Format.cs | header, PBKDF2, AES-256-CTR, HMAC check, payload parser, canonical Huffman decoder |
| Huffman.cs | heap-based Huffman, canonical codes, block encoder |
| Lz.cs | hash-chain match finder, greedy / lazy / DP (fixed costs) / DP with Huffman-derived costs |
| Codec.cs | block codec + NO_COMPRESSION fallback, iterative `dp-iter` parser, defaults found by the GA experiments |
| Knapsack.cs | exact 0/1 DP and greedy |
| Archive.cs | create (analysis, encode, knapsack volume planning, seal) and atomic extract |
| Classifier.cs | perceptron inference (features identical to the Python code, weights from model.json) |
| SelfTest.cs | acceptance tests C1/C2/C5 + reference/bad archives + integrity attack |
| Program.cs | CLI |

## Compatibility check (worth doing once it builds)
The file format is identical to the Python version. Archives from either tool should open in the other:
create with Python, extract with C# (and vice versa). `selftest` already decodes the teacher's `reference/*.arc` and checks
the six `bad_*.arc` error codes.

## Not ported (stay in the Python folder)
perceptron **training**, the genetic algorithm + random/local search, the weak-cipher GA attack, and the report generator.
The C# classifier only loads weights trained by `sarc_cli.py train`. The old `src/` prototype from the original zip is superseded.

## Known differences from the Python code (output is still format-compatible)
- The C# match finder hashes 3-byte prefixes into 16 bits; Python uses exact 3-byte keys. Hash collisions use up the chain depth, so the chosen
  matches (and archive size) can differ by a fraction of a percent. Decoding is unaffected.
- Heap tie-breaking in Huffman differs; code lengths are equally optimal, so sizes are the same up to canonical-tree ties.
