using System.Text;
using System.Text.Json;

namespace Sarc;

/// <summary>Built-in acceptance tests (no test framework needed): the 12 basic cases C1, edge cases C2, security C5, the teacher's reference archives.</summary>
public static class SelfTest
{
    private static int _pass, _fail;
    private static string _root = "";

    private static void Check(string name, Action a)
    {
        try { a(); _pass++; Console.WriteLine($"  PASS  {name}"); }
        catch (Exception e) { _fail++; Console.WriteLine($"  FAIL  {name}: {e.GetType().Name}: {e.Message}"); }
    }

    private static void Assert(bool c, string msg = "assertion failed") { if (!c) throw new Exception(msg); }

    private static void ExpectCode(string code, Action a)
    {
        try { a(); }
        catch (SarcException e) { if (e.Code != code) throw new Exception($"expected {code}, got {e.Code}"); return; }
        throw new Exception($"expected {code}, but no error");
    }

    private static string TempDir()
    {
        string d = Path.Combine(Path.GetTempPath(), "sarc_t_" + Guid.NewGuid().ToString("N")[..10]);
        Directory.CreateDirectory(d); return d;
    }

    private static string MakeDir(Dictionary<string, byte[]> files)
    {
        string d = TempDir();
        foreach (var (n, b) in files) File.WriteAllBytes(Path.Combine(d, n), b);
        return d;
    }

    private static byte[] Rand(int seed, int n) { var r = new Random(seed); var b = new byte[n]; r.NextBytes(b); return b; }

    private static byte[] Repeat(string s, int times) => Encoding.UTF8.GetBytes(string.Concat(Enumerable.Repeat(s, times)));

    private static List<string> Roundtrip(Dictionary<string, byte[]> files, SarcConfig? cfg = null, string pw = "algo2026", ArchiveStats? st = null)
    {
        string src = MakeDir(files), tmp = TempDir();
        var outs = Archiver.Create(src, tmp + "/a", pw, cfg, false, st);
        Archiver.Extract(new[] { tmp + "/a" }, tmp + "/o", pw);
        foreach (var (n, b) in files) Assert(File.ReadAllBytes(Path.Combine(tmp, "o", n)).SequenceEqual(b), "content differs: " + n);
        return outs;
    }

    public static int Run(string? root)
    {
        _root = root ?? ".";
        bool haveRef = Directory.Exists(Path.Combine(_root, "reference")), haveCorpus = Directory.Exists(Path.Combine(_root, "corpus"));
        Console.WriteLine($"SARC self-test (project root: {Path.GetFullPath(_root)}; reference={haveRef}, corpus={haveCorpus})");

        Console.WriteLine("-- primitives");
        Check("CRC32 known vector", () => Assert(Crc32.Compute(Encoding.ASCII.GetBytes("123456789")) == 0xCBF43926u));
        Check("CRC32 incremental", () =>
        {
            var d = Rand(1, 1000);
            Assert(Crc32.Compute(d.AsSpan(500), Crc32.Compute(d.AsSpan(0, 500))) == Crc32.Compute(d));
        });
        Check("C7 canonical codes from lengths", () =>
        {
            var c = Fmt.CanonCodes(new List<(int, int)> { (65, 2), (66, 1), (67, 3), (68, 3) });
            Assert(c[66] == (0UL, 1) && c[65] == (2UL, 2) && c[67] == (6UL, 3) && c[68] == (7UL, 3));
        });
        Check("Huffman lengths optimal (CLRS example = 224 bits) and Kraft sum 1", () =>
        {
            long[] f = { 45, 13, 12, 16, 9, 5 }; var l = Huffman.CodeLengths(f);
            Assert(Enumerable.Range(0, 6).Sum(i => f[i] * l[i]) == 224, "cost");
            Assert(Math.Abs(l.Sum(x => Math.Pow(2, -x)) - 1.0) < 1e-12, "kraft");
        });
        Check("C8 degenerate alphabet lengths", () => { var l = Huffman.CodeLengths(new long[] { 0, 7, 0 }); Assert(l[1] == 1); });
        Check("C10 knapsack DP exact vs brute force, >= greedy", () =>
        {
            var r = new Random(3);
            for (int t = 0; t < 300; t++)
            {
                int n = r.Next(1, 10), W = r.Next(1, 60);
                var v = Enumerable.Range(0, n).Select(_ => (long)r.Next(1, 50)).ToList(); var w = Enumerable.Range(0, n).Select(_ => r.Next(1, 30)).ToList();
                long best = 0;
                for (int m = 0; m < (1 << n); m++)
                {
                    long wt = 0, val = 0;
                    for (int i = 0; i < n; i++) if ((m >> i & 1) != 0) { wt += w[i]; val += v[i]; }
                    if (wt <= W && val > best) best = val;
                }
                var dp = Knapsack.Dp(v, w, W); var gr = Knapsack.Greedy(v, w, W);
                Assert(Knapsack.Total(v, dp) == best, "dp not optimal"); Assert(dp.Sum(i => w[i]) <= W, "overweight");
                Assert(Knapsack.Total(v, gr) <= best);
            }
        });
        Check("C9 greedy/lazy/DP parse: exact expansion, DP cost <= greedy cost", () =>
        {
            var r = new Random(5); const string alpha = "abcab ";
            for (int t = 0; t < 25; t++)
            {
                var a = Enumerable.Range(0, r.Next(1, 600)).Select(_ => (byte)alpha[r.Next(alpha.Length)]).ToArray();
                var g = Lz.ParseGreedy(a, 4096, 32, 3); var z = Lz.ParseLazy(a, 4096, 32, 3, 1); var (d, c) = Lz.ParseDp(a, 4096, 32, 3);
                Assert(Lz.Expand(g).SequenceEqual(a) && Lz.Expand(z).SequenceEqual(a) && Lz.Expand(d).SequenceEqual(a), "expand");
                Assert(c == Lz.TokensCost(d) && c <= Lz.TokensCost(g), "dp cost");
            }
        });
        Check("block codec roundtrip (all strategies, edge inputs)", () =>
        {
            var inputs = new List<byte[]> { Array.Empty<byte>(), new byte[] { 65 }, Enumerable.Repeat((byte)65, 5000).ToArray(), Repeat("hello world ", 500), Rand(2, 3000) };
            foreach (var inp in inputs)
                foreach (int s in new[] { Strategy.Greedy, Strategy.Lazy, Strategy.Dp })
                {
                    var p = new CodecParams { Strategy = s };
                    var (flag, coded) = Codec.EncodeBlock(inp, p);
                    if (flag == 1) { Assert(coded.SequenceEqual(inp)); continue; }
                    var o = new ByteBuf(); Fmt.DecodeBlock(coded, inp.Length, o); Assert(o.ToArray().SequenceEqual(inp), "decode mismatch");
                }
        });

        Console.WriteLine("-- C1 acceptance cases");
        Check("C1 single text file", () => Roundtrip(new() { ["a.txt"] = Repeat("hello world, hello archive! ", 40) }));
        Check("C2 empty file and 1-byte file", () => Roundtrip(new() { ["e.dat"] = Array.Empty<byte>(), ["o.dat"] = new byte[] { 120 } }));
        Check("C4 identical bytes shrink >5x", () =>
        {
            var st = new ArchiveStats(); Roundtrip(new() { ["z.dat"] = new byte[20000] }, null, "algo2026", st);
            Assert(20000.0 / st.ArcBytes > 5, $"ratio {20000.0 / st.ArcBytes:F2}");
        });
        Check("C5 noise: NO_COMPRESSION, coded_len = raw_len, measured overhead", () =>
        {
            var noise = Rand(1, 10000); string src = MakeDir(new() { ["n.bin"] = noise }), tmp = TempDir(); var st = new ArchiveStats();
            Archiver.Create(src, tmp + "/a", "algo2026", null, false, st);
            var (_, pt) = Fmt.OpenVolume(File.ReadAllBytes(tmp + "/a.arc"), "algo2026"); var (_, bl) = Fmt.ParsePayload(pt);
            Assert(bl.All(b => b.Flag == 1 && b.Coded == b.Raw), "not stored");
            Assert(st.ArcBytes - noise.Length == Fmt.HDR + Fmt.TAG + 6 + (2 + 5 + 10) + 15, "overhead " + (st.ArcBytes - noise.Length));
        });
        Check("C11 tamper / wrong password / truncation", () =>
        {
            string src = MakeDir(new() { ["a.txt"] = Repeat("secret data ", 100) }), tmp = TempDir();
            Archiver.Create(src, tmp + "/a", "algo2026"); var d = File.ReadAllBytes(tmp + "/a.arc");
            var b = (byte[])d.Clone(); b[50] ^= 1;
            ExpectCode("E_HMAC", () => Fmt.OpenVolume(b, "algo2026"));
            ExpectCode("E_HMAC", () => Fmt.OpenVolume(d, "wrong"));
            foreach (int cut in new[] { 0, 3, 40, 41, 72, d.Length - 1 }) { int c = cut; ExpectCode("E_LENGTH", () => Fmt.OpenVolume(d[..c], "algo2026")); }
        });
        Check("C12 volumes: any order, missing, duplicate, big file split into blocks", () =>
        {
            var big = Rand(9, 70000).Concat(Repeat("tail", 3000)).ToArray();
            string src = MakeDir(new() { ["big.bin"] = big, ["a.txt"] = Repeat("abc ", 3000) }), tmp = TempDir();
            var outs = Archiver.Create(src, tmp + "/a", "algo2026", new SarcConfig { MaxVolume = 32768 });
            Assert(outs.Count > 1, "expected several volumes");
            Assert(outs.All(p => new FileInfo(p).Length <= 32768), "volume limit");
            var shuffled = outs.AsEnumerable().Reverse().ToList();
            Archiver.Extract(shuffled, tmp + "/o", "algo2026");
            Assert(File.ReadAllBytes(tmp + "/o/big.bin").SequenceEqual(big));
            ExpectCode("E_VOLUME_MISSING", () => Archiver.Extract(outs.Skip(1).ToList(), tmp + "/o2", "algo2026"));
            ExpectCode("E_VOLUME_DUP", () => Archiver.Extract(outs.Append(outs[0]).ToList(), tmp + "/o3", "algo2026"));
            Assert(!Directory.Exists(tmp + "/o2") || Directory.GetFileSystemEntries(tmp + "/o2").Length == 0, "partial output");
        });

        if (haveRef)
        {
            Console.WriteLine("-- teacher's reference archives");
            string R = Path.Combine(_root, "reference");
            foreach (var n in new[] { "ref_01_single_file", "ref_02_empty_file", "ref_03_degenerate", "ref_04_seven_groups", "ref_05_no_compression", "ref_06_two_volumes" })
            {
                string name = n;
                Check("reference " + name, () =>
                {
                    using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(R, name + ".json")));
                    var expect = doc.RootElement.GetProperty("вміст");
                    var got = Archiver.DecodeArchive(Archiver.FindVolumes(new[] { Path.Combine(R, name) }), "algo2026");
                    Assert(got.Count == expect.EnumerateObject().Count(), "file count");
                    foreach (var p in expect.EnumerateObject())
                        Assert(Crc32.Compute(got[p.Name]) == Convert.ToUInt32(p.Value.GetProperty("crc32").GetString(), 16), "crc " + p.Name);
                });
            }
            using var ef = JsonDocument.Parse(File.ReadAllText(Path.Combine(R, "expected_failures.json")));
            foreach (var p in ef.RootElement.EnumerateObject())
            {
                string file = p.Name, code = p.Value.GetProperty("очікуваний_код").GetString()!;
                Check($"bad archive {file} -> {code}", () => ExpectCode(code, () => Archiver.DecodeArchive(new[] { Path.Combine(R, file) }, "algo2026")));
            }
        }
        else Console.WriteLine("-- (reference/ not found: pass the project root as argument to run reference tests)");

        if (haveCorpus)
        {
            Console.WriteLine("-- C3 corpus");
            Check("C3 all seven groups restored byte-exact, volumes <= 65536, raw size 329903", () =>
            {
                string corp = Path.Combine(_root, "corpus"); var files = new Dictionary<string, byte[]>();
                foreach (var f in Directory.GetFiles(corp)) if (Path.GetFileName(f) != "labels.csv") files[Path.GetFileName(f)] = File.ReadAllBytes(f);
                var st = new ArchiveStats(); string model = Path.Combine(AppContext.BaseDirectory, "model.json");
                var cfg = new SarcConfig(); if (File.Exists(model)) cfg.Classifier = new PerceptronClassifier(model).Classify;
                var outs = Roundtrip(files, cfg, "algo2026", st);
                Assert(st.OrigBytes == 329903, "orig " + st.OrigBytes); Assert(outs.All(p => new FileInfo(p).Length <= 65536), "volume size");
                Console.WriteLine($"        raw={st.OrigBytes} archive={st.ArcBytes} ratio={(double)st.OrigBytes / st.ArcBytes:F3} volumes={st.Volumes}");
            });
            Check("C6 already-compressed group stored without compression", () =>
            {
                string corp = Path.Combine(_root, "corpus"); string model = Path.Combine(AppContext.BaseDirectory, "model.json");
                var cfg = new SarcConfig(); if (File.Exists(model)) cfg.Classifier = new PerceptronClassifier(model).Classify;
                string src = MakeDir(new() { ["f0005.z"] = File.ReadAllBytes(Path.Combine(corp, "f0005.z")) }), tmp = TempDir();
                Archiver.Create(src, tmp + "/a", "algo2026", cfg);
                var (_, pt) = Fmt.OpenVolume(File.ReadAllBytes(tmp + "/a.arc"), "algo2026"); Assert(Fmt.ParsePayload(pt).blocks.All(b => b.Flag == 1));
            });
        }

        Console.WriteLine("-- C2 edge cases / C5 security");
        Check("input validation", () =>
        {
            string tmp = TempDir();
            ExpectCode("E_INPUT", () => Archiver.Create(tmp + "/nope", tmp + "/a", "pw"));
            ExpectCode("E_INPUT", () => Archiver.Create(MakeDir(new()), tmp + "/a", "pw"));
            ExpectCode("E_INPUT", () => Archiver.Create(MakeDir(new() { ["a"] = new byte[] { 1 } }), tmp + "/a", ""));
            int old = Archiver.MaxNameBytes; Archiver.MaxNameBytes = 10;
            try { ExpectCode("E_INPUT", () => Archiver.Create(MakeDir(new() { [new string('x', 11)] = new byte[] { 1 } }), tmp + "/a", "pw")); }
            finally { Archiver.MaxNameBytes = old; }
        });
        Check("file shorter than min match", () => Roundtrip(new() { ["t.txt"] = new byte[] { 97, 98 } }));
        Check("volume smaller than header -> controlled E_CAPACITY", () =>
        {
            string src = MakeDir(new() { ["a.txt"] = Repeat("hello ", 50) }), tmp = TempDir();
            ExpectCode("E_CAPACITY", () => Archiver.Create(src, tmp + "/a", "pw", new SarcConfig { MaxVolume = 60 }));
            ExpectCode("E_CAPACITY", () => Archiver.Create(src, tmp + "/a", "pw", new SarcConfig { MaxVolume = 100 }));
        });
        Check("volume limit 10x smaller", () =>
        {
            var cfg = new SarcConfig { MaxVolume = 6554 }; cfg.Params.BlockSize = 4096;
            var outs = Roundtrip(new() { ["a.txt"] = Repeat("lorem ipsum dolor sit amet ", 800), ["b.bin"] = Rand(4, 9000) }, cfg);
            Assert(outs.Count >= 3 && outs.All(p => new FileInfo(p).Length <= 6554));
        });
        Check("unsafe names rejected", () =>
        {
            foreach (var bad in new[] { "../x", "/abs", "a\\b", "c:x", "..", ".", "a\0b", "" }) Assert(!Fmt.SafeName(bad), bad);
            var salt = new byte[16]; var (ke, km) = Fmt.Derive(Encoding.UTF8.GetBytes("algo2026"), salt);
            var evil = new List<byte> { 0, 1, 0, 0, 0, 3, (byte)'.', (byte)'.', (byte)'/', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            foreach (var pt in new[] { Array.Empty<byte>(), new byte[] { 0, 1, 0, 0 }, Enumerable.Repeat((byte)0xFF, 50).ToArray(), evil.ToArray() })
            {
                var vol = Archiver.SealVolume(pt, ke, km, salt, new byte[8], 0, 1);
                ExpectCode("E_PAYLOAD", () => Fmt.ParsePayload(Fmt.OpenVolume(vol, "algo2026").payload));
            }
        });
        Check("atomic extraction keeps existing files on failure", () =>
        {
            string src = MakeDir(new() { ["a.txt"] = Repeat("new content ", 50) }), tmp = TempDir();
            Archiver.Create(src, tmp + "/a", "algo2026"); Directory.CreateDirectory(tmp + "/o"); File.WriteAllText(tmp + "/o/a.txt", "OLD");
            ExpectCode("E_HMAC", () => Archiver.Extract(new[] { tmp + "/a" }, tmp + "/o", "wrongpw"));
            Assert(File.ReadAllText(tmp + "/o/a.txt") == "OLD" && Directory.GetFileSystemEntries(tmp + "/o").Length == 1);
        });
        Check("public part leaks neither names nor password; fresh salt per archive; distinct nonce per volume", () =>
        {
            string src = MakeDir(new() { ["secret_name.txt"] = Rand(2, 40000) }), tmp = TempDir();
            var o1 = Archiver.Create(src, tmp + "/a", "algo2026", new SarcConfig { MaxVolume = 16384 });
            var o2 = Archiver.Create(src, tmp + "/b", "algo2026", new SarcConfig { MaxVolume = 16384 });
            var d = File.ReadAllBytes(o1[0]); Assert(Encoding.UTF8.GetString(d).IndexOf("secret_name", StringComparison.Ordinal) < 0);
            var h1 = o1.Select(p => Fmt.ParseHeader(File.ReadAllBytes(p))).ToList(); var h2 = Fmt.ParseHeader(File.ReadAllBytes(o2[0]));
            Assert(h1.Select(h => Convert.ToHexString(h.Salt)).Distinct().Count() == 1 && !h1[0].Salt.SequenceEqual(h2.Salt));
            Assert(h1.Select(h => Convert.ToHexString(h.Nonce)).Distinct().Count() == h1.Count, "nonce reuse");
        });
        Check("integrity: single-bit modifications are all rejected", () =>
        {
            string src = MakeDir(new() { ["a.txt"] = Repeat("abc def ", 200) }), tmp = TempDir();
            Archiver.Create(src, tmp + "/a", "algo2026"); Assert(IntegrityAttack(File.ReadAllBytes(tmp + "/a.arc"), "algo2026", true) == 0);
        });

        Console.WriteLine($"\n{_pass} passed, {_fail} failed");
        return _fail == 0 ? 0 : 1;
    }

    /// <summary>Attack team's integrity check: flip one bit at many positions; the decoder must reject EVERY modification.</summary>
    public static int IntegrityAttack(byte[] archive, string password, bool quiet = false)
    {
        var r = new Random(5); int L = archive.Length;
        var pos = new SortedSet<int> { 0, 4, 5, 9, 25, 37, 41, L - 33, L - 32, L - 1 };
        for (int i = 0; i < 60; i++) pos.Add(r.Next(L));
        int accepted = 0, tried = 0;
        foreach (int p in pos.Where(q => q >= 0 && q < L))
        {
            var b = (byte[])archive.Clone(); b[p] ^= 1; tried++;
            try { Fmt.OpenVolume(b, password); accepted++; Console.WriteLine($"  byte {p}: ACCEPTED (BUG)"); }
            catch (SarcException e) { if (!quiet && tried <= 12) Console.WriteLine($"  byte {p}: {e.Code}"); }
        }
        Console.WriteLine($"{tried} single-bit modifications tried; accepted by the decoder: {accepted}");
        return accepted == 0 ? 0 : 1;
    }
}
