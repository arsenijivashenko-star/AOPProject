# SarcArchiver — C# (.NET 9) port

**Status: written but NOT compiled or run.** The environment that produced this had no .NET SDK, so expect to fix a few
compile errors on the first `dotnet build`. The logic is a line-by-line port of the Python version, which *was* tested
(28 tests, all reference archives). Please run `selftest` first; it will tell you quickly what works.

## Build / run
```
dotnet build -c Release
dotnet run -c Release -- selftest /path/to/Project        # Project/ = folder with corpus/ and reference/
dotnet run -c Release -- archive /path/to/corpus out/c -p algo2026
dotnet run -c Release -- extract out/c out/restored -p algo2026
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
| Lz.cs | hash-chain match finder, greedy / lazy / DP (optimal) parsers |
| Codec.cs | block codec + NO_COMPRESSION fallback, defaults found by the GA experiments |
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
