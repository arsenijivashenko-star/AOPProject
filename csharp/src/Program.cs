using System.Text;

namespace Sarc;

public static class Program
{
    private const string Usage = @"SARC smart secure archiver (C#)
  SarcArchiver archive INPUT OUT_BASE [-p PASSWORD] [--max-volume N] [--block-size N] [--strategy greedy|lazy|dp|dp-iter]
                                      [--packer dp|greedy] [--recursive] [--no-classifier] [--model model.json]
  SarcArchiver extract ARCHIVE [ARCHIVE...] OUTDIR [-p PASSWORD]     (ARCHIVE: base name or .arc volume files, any order)
  SarcArchiver list ARCHIVE [ARCHIVE...] [-p PASSWORD]                (file table: names, sizes, CRC32, type, strategy, volumes)
  SarcArchiver verify ARCHIVE [ARCHIVE...] [-p PASSWORD]              (full integrity check, writes nothing)
  SarcArchiver info VOLUME.arc [...]                                  (public header only, nothing is decrypted)
  SarcArchiver attack integrity ARCHIVE.arc [-p PASSWORD]             (flips single bits; every change must be rejected)
  SarcArchiver selftest [PROJECT_ROOT]                                (acceptance tests; PROJECT_ROOT holds corpus/ reference/ ...)
Password: -p, or env SARC_PASSWORD, or prompt.  Exit codes: 0 ok, 1 test/attack failure, 2 archive error, 3 I/O error, 4 internal error.";

    public static int Main(string[] args)
    {
        try { return Run(args); }
        catch (SarcException e) { Console.Error.WriteLine($"error {e.Code}: {e.Message}"); return 2; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or OutOfMemoryException)
        { Console.Error.WriteLine($"error E_IO: {e.GetType().Name}: {e.Message}"); return 3; }
        catch (Exception e)
        { Console.Error.WriteLine($"error E_INTERNAL: {e.GetType().Name}: {e.Message}"); return 4; }
    }

    private sealed class Opts
    {
        public List<string> Pos = new(); public Dictionary<string, string> Val = new(); public HashSet<string> Flag = new();
    }

    private static Opts Parse(string[] a, int from)
    {
        var o = new Opts();
        var withValue = new HashSet<string> { "-p", "--password", "--max-volume", "--block-size", "--strategy", "--packer", "--model" };
        for (int i = from; i < a.Length; i++)
        {
            if (a[i].StartsWith('-') && a[i].Length > 1)
            {
                if (withValue.Contains(a[i])) { if (i + 1 >= a.Length) throw new SarcException("E_INPUT", $"missing value for {a[i]}"); o.Val[a[i]] = a[++i]; }
                else o.Flag.Add(a[i]);
            }
            else o.Pos.Add(a[i]);
        }
        return o;
    }

    private static string Password(Opts o)
    {
        if (o.Val.TryGetValue("-p", out var p) || o.Val.TryGetValue("--password", out p)) return p;
        var env = Environment.GetEnvironmentVariable("SARC_PASSWORD");
        if (!string.IsNullOrEmpty(env)) return env;
        Console.Error.Write("Password: ");
        if (Console.IsInputRedirected) return Console.ReadLine() ?? "";
        var sb = new StringBuilder();
        while (true)
        {
            var k = Console.ReadKey(true);
            if (k.Key == ConsoleKey.Enter) break;
            if (k.Key == ConsoleKey.Backspace) { if (sb.Length > 0) sb.Length--; }
            else sb.Append(k.KeyChar);
        }
        Console.Error.WriteLine();
        return sb.ToString();
    }

    private static int IntOpt(Opts o, string key, int def)
        => o.Val.TryGetValue(key, out var v) ? (int.TryParse(v, out int r) ? r : throw new SarcException("E_INPUT", $"bad number for {key}")) : def;

    public static int Run(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help") { Console.WriteLine(Usage); return args.Length == 0 ? 1 : 0; }
        var o = Parse(args, 1);
        switch (args[0])
        {
            case "archive":
            {
                if (o.Pos.Count != 2) throw new SarcException("E_INPUT", "usage: archive INPUT OUT_BASE");
                var cfg = new SarcConfig { MaxVolume = IntOpt(o, "--max-volume", 65536), Packer = o.Val.GetValueOrDefault("--packer", "dp") };
                cfg.Params.BlockSize = IntOpt(o, "--block-size", 16384);
                cfg.Params.Strategy = o.Val.GetValueOrDefault("--strategy", "lazy") switch
                {
                    "greedy" => Strategy.Greedy, "lazy" => Strategy.Lazy, "dp" => Strategy.Dp, "dp-iter" => Strategy.Dp2,
                    _ => throw new SarcException("E_INPUT", "strategy must be greedy|lazy|dp|dp-iter")
                };
                if (cfg.Packer != "dp" && cfg.Packer != "greedy") throw new SarcException("E_INPUT", "packer must be dp|greedy");
                string model = o.Val.GetValueOrDefault("--model", Path.Combine(AppContext.BaseDirectory, "model.json"));
                if (!o.Flag.Contains("--no-classifier") && File.Exists(model)) cfg.Classifier = new PerceptronClassifier(model).Classify;
                var st = new ArchiveStats();
                var outs = Archiver.Create(o.Pos[0], o.Pos[1], Password(o), cfg, o.Flag.Contains("--recursive"), st);
                foreach (var p in outs) Console.WriteLine($"wrote {p} ({new FileInfo(p).Length} bytes)");
                Console.WriteLine($"files={st.Files.Count} orig={st.OrigBytes} archive={st.ArcBytes} ratio={(double)st.OrigBytes / Math.Max(1, st.ArcBytes):F3} volumes={st.Volumes} time={st.Seconds:F2}s");
                return 0;
            }
            case "extract":
            {
                if (o.Pos.Count < 2) throw new SarcException("E_INPUT", "usage: extract ARCHIVE... OUTDIR");
                var outDir = o.Pos[^1]; var arcs = o.Pos.Take(o.Pos.Count - 1).ToList();
                var names = Archiver.Extract(arcs, outDir, Password(o));
                Console.WriteLine($"extracted {names.Count} file(s) to {outDir}: all CRC32 verified");
                return 0;
            }
            case "list":
            {
                if (o.Pos.Count < 1) throw new SarcException("E_INPUT", "usage: list ARCHIVE...");
                var (files, tot) = Archiver.ListArchive(o.Pos, Password(o));
                Console.WriteLine($"{files.Count} file(s), {tot} volume(s)");
                foreach (var f in files)
                    Console.WriteLine($"  {f.Name,-32} {f.Size,9} B  crc32={f.Crc:x8}  type={f.Type}  strat={Strategy.Name(f.Strat)}  vol=[{string.Join(", ", f.Volumes)}]");
                return 0;
            }
            case "verify":
            {
                if (o.Pos.Count < 1) throw new SarcException("E_INPUT", "usage: verify ARCHIVE...");
                int n = Archiver.Verify(o.Pos, Password(o));
                Console.WriteLine($"OK: {n} file(s) verified (HMAC, structure, CRC32); nothing written");
                return 0;
            }
            case "info":
            {
                foreach (var p in o.Pos)
                {
                    try
                    {
                        var d = File.ReadAllBytes(p); var h = Fmt.ParseHeader(d);
                        Console.WriteLine($"{p}: SARC v1 volume {h.Idx}/{h.Tot} ciphertext={h.N}B total={d.Length}B salt={Convert.ToHexString(h.Salt).ToLowerInvariant()} (public)");
                    }
                    catch (SarcException e) { Console.WriteLine($"{p}: INVALID ({e.Code})"); }
                }
                return 0;
            }
            case "attack":
            {
                if (o.Pos.Count != 2 || o.Pos[0] != "integrity") throw new SarcException("E_INPUT", "usage: attack integrity ARCHIVE.arc  (GA cryptanalysis of weak ciphers lives in the Python tools)");
                return SelfTest.IntegrityAttack(File.ReadAllBytes(o.Pos[1]), Password(o));
            }
            case "selftest":
                return SelfTest.Run(o.Pos.Count > 0 ? o.Pos[0] : null);
            default:
                Console.Error.WriteLine(Usage); return 1;
        }
    }
}
