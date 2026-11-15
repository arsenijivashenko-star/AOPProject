namespace Sarc;

/// <summary>CRC-32 (IEEE 802.3), own table-driven implementation. Incremental: Compute(b2, Compute(b1)).</summary>
public static class Crc32
{
    private static readonly uint[] Table = Make();

    private static uint[] Make()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }

    public static uint Compute(ReadOnlySpan<byte> data, uint crc = 0)
    {
        uint c = crc ^ 0xFFFFFFFFu;
        foreach (byte b in data) c = Table[(c ^ b) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }
}
