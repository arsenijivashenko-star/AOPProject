namespace Sarc;

/// <summary>Controlled archive error. Code is one of E_MAGIC, E_VERSION, E_LENGTH, E_HMAC, E_PAYLOAD, E_CRC,
/// E_VOLUME_MISSING, E_VOLUME_DUP, E_CAPACITY, E_INPUT.</summary>
public sealed class SarcException : Exception
{
    public string Code { get; }
    public SarcException(string code, string msg = "") : base($"{code}: {msg}") { Code = code; }
}
