# META-USAGE
Code in this delivery was written with AI assistance (Claude, Anthropic) at the user's request, in one session.
- Format reverse-engineered from reference/*.arc (FORMAT_SPEC_P2.md was NOT in the zip); verified against all reference and bad archives.
- Huffman, LZ77 (greedy/lazy/DP), knapsack, perceptron (numpy matrix maths only), GA, attacks, CRC32 are own implementations.
- External libraries: `cryptography` (AES-CTR, PBKDF2, SHA-256/HMAC only), numpy, matplotlib (plot). No zlib/lzma in the path (zlib only in tests as CRC reference).
- The team must review, understand and be able to defend this code (E1/E2); attribute it as required by the course rules.
- 16.11.2026: profiles, `--only`, `--dry-run`, `--json`, `--progress`, `--password-file`, CLI error handling, new tests and ROADMAP section 2.1 were added with AI assistance (Claude); the author must review them.
- 16.11.2026 (second session): `--jobs` (parallel block encoding), `--include/--exclude`, `diff`, `bench`, `pyproject.toml`, edge-case tests and ROADMAP section 2.2 were added with AI assistance (Claude); the author must review them.
- Selective extraction (`extract --only`) added during the current improvement pass: filters decoded block data by exact archived file name while still authenticating and structurally parsing every supplied volume.
