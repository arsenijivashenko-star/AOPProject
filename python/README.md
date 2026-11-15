# SARC smart secure archiver (Python reference implementation)
Run from this folder (needs python3.10+, `pip install cryptography numpy matplotlib`).
```
python3 sarc_cli.py archive INPUT_DIR OUT_BASE -p PASSWORD [--max-volume 65536] [--strategy lazy|greedy|dp|dp-iter] [--recursive]
python3 sarc_cli.py extract OUT_BASE OUTDIR -p PASSWORD      # base name, .arc file(s) in any order
python3 sarc_cli.py list OUT_BASE -p PASSWORD                  # file table (names, sizes, CRC32, strategy, volumes)
python3 sarc_cli.py verify OUT_BASE -p PASSWORD                # full integrity check, writes nothing
python3 sarc_cli.py info OUT_BASE.part0.arc
python3 sarc_cli.py train PROJECT_ROOT                       # retrain perceptron (train/ + valid/ -> model.json)
python3 sarc_cli.py attack integrity OUT_BASE.arc -p PASSWORD
python3 sarc_cli.py attack weak FILE --cipher xor|vigenere --data PROJECT_ROOT --out recovered.bin
python3 run_ga.py 40 ; python3 report.py ; python3 iter_report.py ..                    # GA experiment, report tables
SARC_PROJECT=/path/to/Project python3 -m unittest -v tests/test_sarc.py
```
Errors: E_MAGIC E_VERSION E_LENGTH E_HMAC E_PAYLOAD E_CRC E_VOLUME_MISSING E_VOLUME_DUP E_CAPACITY E_INPUT.
See REPORT_TABLES.md for measured results and NOTES below.

## Format (inferred from the reference archives)
Header 41 B (SARC, ver 1, vol idx u16, total u16, salt16, nonce12, ct_len u32; big-endian) | AES-256-CTR ciphertext | HMAC-SHA256(header+ct).
Payload: nfiles u16, reserved u16 (=0), per file {name_len u16, name, size u32, crc32 u32, type u8, strategy u8}, nblocks u16,
per block {file_idx u16, block_index u32, raw_len u32, coded_len u32, flag u8 (0 Huffman/LZ, 1 stored)}, then block data.
Coded block: 3 canonical-Huffman trees (u16 count, {sym u16, len u8}) for literals+marker 256 / length-3 / distance slot (bit_length(dist-1), then slot-1 extra bits), u32 tokens, u32 bytes, bitstream.
Keys: PBKDF2-HMAC-SHA256 100k it, 64 B -> K_enc | K_mac, once per archive. Nonce = 8 random bytes || u32 volume index.

## Notes / honest limitations
- Implemented in Python, not in the C# prototype (no .NET available to build/test here). The old C# in src/ was a non-working draft.
- GA: no significant gain over random/local search at budget 40 (all ~0.796 Q; differences below timing noise). Landscape is flat; fitness uses wall time.
- Weak-cipher attack: XOR and Vigenere keys recovered exactly in tests; substitution over the full byte alphabet is NOT solved (experimental).
- `dp-iter` strategy (iterative DP with Huffman-derived costs, REPORT_TABLES.md §7) is the densest parser: ~1.4% below fixed-cost DP, ~2.7% below lazy on the corpus; ~5x slower than lazy.
- Not implemented: metadata attack, arithmetic coding, byte predictor, DH/RSA, block-size alignment, DP block splitting, streaming of coded blocks (coded blocks are held in RAM; input read per block).
- Perceptron macro-F1 on valid/: 0.9905 (F1_min 0.61). holdout/ not available.
- Knapsack W uses 41+32+6+table and weight=coded+15 (the "-15*blocks" term in team_config would double-count).
