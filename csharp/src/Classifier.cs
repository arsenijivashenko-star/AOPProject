using System.Text.Json;

namespace Sarc;

/// <summary>File-type perceptron (inference). The network was trained on train/ and selected on valid/ (see README); weights are loaded
/// from model.json. Features: Shannon entropy, bigram entropy, printable share, newline spacing, 16 autocorrelation lags, 32-bin byte histogram, ...</summary>
public sealed class PerceptronClassifier
{
    public const int NFeat = 61, Sample = 16384;
    private readonly double[] _mu, _sd;
    private readonly double[][][] _w;      // layer -> in -> out
    private readonly double[][] _b;

    public PerceptronClassifier(string modelPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(modelPath));
        var root = doc.RootElement;
        _mu = root.GetProperty("mu").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        _sd = root.GetProperty("sd").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        var net = root.GetProperty("net");
        _w = net.GetProperty("W").EnumerateArray()
            .Select(layer => layer.EnumerateArray().Select(row => row.EnumerateArray().Select(e => e.GetDouble()).ToArray()).ToArray()).ToArray();
        _b = net.GetProperty("b").EnumerateArray().Select(l => l.EnumerateArray().Select(e => e.GetDouble()).ToArray()).ToArray();
        if (_mu.Length != NFeat || _sd.Length != NFeat || _w[0].Length != NFeat) throw new InvalidDataException("model.json does not match the feature vector");
    }

    public static double[] Features(byte[] data)
    {
        int n = Math.Min(data.Length, Sample);
        var f = new double[NFeat];
        if (n == 0) return f;
        var cnt = new long[256];
        for (int i = 0; i < n; i++) cnt[data[i]]++;
        double H = 0; int uniqueSyms = 0;
        for (int c = 0; c < 256; c++) if (cnt[c] > 0) { double p = cnt[c] / (double)n; H -= p * Math.Log2(p); uniqueSyms++; }
        H /= 8.0;
        long printable = 0, ctrl = 0, high = 0, alpha = 0, punct = 0;
        long firstNl = -1, lastNl = -1, nlCount = 0; double sum = 0;
        const string punctSet = "(){}[]=:;,\"_#<>";
        for (int i = 0; i < n; i++)
        {
            byte a = data[i]; sum += a;
            if ((a >= 32 && a < 127) || a == 10 || a == 9) printable++;
            if (a < 9) ctrl++;
            if (a >= 128) high++;
            if ((a >= 65 && a <= 90) || (a >= 97 && a <= 122)) alpha++;
            if (punctSet.IndexOf((char)a) >= 0 && a < 128) punct++;
            if (a == 10) { if (firstNl < 0) firstNl = i; lastNl = i; nlCount++; }
        }
        double nlgap = nlCount > 1 ? (double)(lastNl - firstNl) / (nlCount - 1) : 0.0;
        nlgap = Math.Min(nlgap, 256.0) / 256.0;
        double mean = sum / n, variance = 0;
        for (int i = 0; i < n; i++) { double x = data[i] - mean; variance += x * x; }
        var ac = new double[16];
        for (int lag = 1; lag <= 16; lag++)
        {
            if (variance > 0 && n > lag)
            {
                double s = 0;
                for (int i = 0; i + lag < n; i++) s += (data[i] - mean) * (data[i + lag] - mean);
                ac[lag - 1] = s / variance;
            }
        }
        double Hb = 0;
        if (n > 1)
        {
            var bg = new int[65536];
            for (int i = 0; i + 1 < n; i++) bg[(data[i] << 8) | data[i + 1]]++;
            for (int k = 0; k < bg.Length; k++) if (bg[k] > 0) { double p = bg[k] / (double)(n - 1); Hb -= p * Math.Log2(p); }
            Hb /= 16.0;
        }
        int o = 0;
        f[o++] = H; f[o++] = Hb; f[o++] = printable / (double)n; f[o++] = nlgap; f[o++] = nlCount / (double)n;
        f[o++] = ctrl / (double)n; f[o++] = high / (double)n; f[o++] = cnt[0] / (double)n; f[o++] = uniqueSyms / 256.0;
        f[o++] = alpha / (double)n; f[o++] = punct / (double)n; f[o++] = cnt[44] / (double)n; f[o++] = cnt[32] / (double)n;
        for (int k = 0; k < 16; k++) f[o++] = ac[k];
        for (int k = 0; k < 32; k++) { double s = 0; for (int j = 0; j < 8; j++) s += cnt[k * 8 + j]; f[o++] = s / n; }
        return f;
    }

    public int Classify(byte[] data)
    {
        double[] f = Features(data);
        double[] a = new double[NFeat];
        for (int i = 0; i < NFeat; i++) a[i] = (f[i] - _mu[i]) / _sd[i];
        for (int l = 0; l < _w.Length; l++)
        {
            int outN = _b[l].Length; var z = new double[outN];
            for (int o = 0; o < outN; o++) z[o] = _b[l][o];
            for (int i = 0; i < a.Length; i++) { double ai = a[i]; if (ai == 0) continue; var row = _w[l][i]; for (int o = 0; o < outN; o++) z[o] += ai * row[o]; }
            if (l < _w.Length - 1) for (int o = 0; o < outN; o++) z[o] = Math.Max(0, z[o]);
            a = z;
        }
        int best = 0;
        for (int k = 1; k < a.Length; k++) if (a[k] > a[best]) best = k;
        return best;
    }
}
