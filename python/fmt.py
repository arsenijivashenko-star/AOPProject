"""SARC format: reader (volume decode) verified against reference/ archives."""
import hmac, hashlib, struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

class SarcError(Exception):
    def __init__(self, code, msg=''):
        super().__init__(f'{code}: {msg}'); self.code = code

HDR = 41; TAG = 32

def derive(pw: bytes, salt: bytes):
    k = PBKDF2HMAC(algorithm=hashes.SHA256(), length=64, salt=salt, iterations=100000).derive(pw)
    return k[:32], k[32:]

def ctr(key, nonce, data):
    return Cipher(algorithms.AES(key), modes.CTR(nonce + (1).to_bytes(4, 'big'))).encryptor().update(data)

def _mk_table():
    t = []
    for n in range(256):
        c = n
        for _ in range(8): c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
        t.append(c)
    return t
_CRC = _mk_table()

def crc32(b, crc=0):
    """CRC-32 (IEEE), own table-driven implementation; incremental: crc32(b2, crc32(b1))."""
    c = crc ^ 0xFFFFFFFF; T = _CRC
    for x in b: c = T[(c ^ x) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF

def safe_name(n: str) -> bool:
    if not n or '/' in n or '\\' in n or ':' in n or '\0' in n: return False
    if n in ('.', '..') or n.startswith('/'): return False
    return True

def parse_header(d: bytes):
    if len(d) < HDR + TAG: raise SarcError('E_LENGTH', 'too short')
    if d[:4] != b'SARC': raise SarcError('E_MAGIC')
    ver = d[4]
    if ver != 1: raise SarcError('E_VERSION')
    idx, tot = struct.unpack('>HH', d[5:9]); salt = d[9:25]; nonce = d[25:37]
    n = struct.unpack('>I', d[37:41])[0]
    if HDR + n + TAG != len(d): raise SarcError('E_LENGTH', f'{HDR+n+TAG} != {len(d)}')
    if tot == 0 or idx >= tot: raise SarcError('E_PAYLOAD', 'volume index')
    return dict(idx=idx, tot=tot, salt=salt, nonce=nonce, n=n)

def open_volume(d: bytes, pw: str, keys_cache=None):
    h = parse_header(d)             # validated BEFORE any decryption
    key = h['salt']
    if keys_cache is not None and key in keys_cache: ke, km = keys_cache[key]
    else:
        ke, km = derive(pw.encode('utf-8'), h['salt'])
        if keys_cache is not None: keys_cache[key] = (ke, km)
    tag = hmac.new(km, d[:HDR + h['n']], hashlib.sha256).digest()
    if not hmac.compare_digest(tag, d[HDR + h['n']:]): raise SarcError('E_HMAC')
    return h, ctr(ke, h['nonce'], d[HDR:HDR + h['n']])

# ---------- payload ----------
def parse_payload(pt: bytes):
    try:
        o = 0
        nfiles, resv = struct.unpack_from('>HH', pt, 0); o = 4
        if resv != 0: raise SarcError('E_PAYLOAD', 'reserved != 0')
        files = []
        for _ in range(nfiles):
            (nl,) = struct.unpack_from('>H', pt, o); o += 2
            raw = pt[o:o + nl]
            if len(raw) != nl: raise SarcError('E_PAYLOAD', 'name')
            o += nl
            name = raw.decode('utf-8')
            size, crc, typ, strat = struct.unpack_from('>IIBB', pt, o); o += 10
            if not safe_name(name): raise SarcError('E_PAYLOAD', 'unsafe name')
            files.append(dict(name=name, size=size, crc=crc, type=typ, strat=strat))
        (nblocks,) = struct.unpack_from('>H', pt, o); o += 2
        blocks = []
        for _ in range(nblocks):
            fi, off, raw_len, coded_len, flag = struct.unpack_from('>HIIIB', pt, o); o += 15
            if fi >= nfiles: raise SarcError('E_PAYLOAD', 'file idx')
            blocks.append(dict(fi=fi, off=off, raw=raw_len, coded=coded_len, flag=flag))
        for b in blocks:
            b['data'] = pt[o:o + b['coded']]
            if len(b['data']) != b['coded']: raise SarcError('E_PAYLOAD', 'block data')
            o += b['coded']
        if o != len(pt): raise SarcError('E_PAYLOAD', 'trailing')
        return files, blocks
    except (struct.error, UnicodeDecodeError) as e:
        raise SarcError('E_PAYLOAD', str(e))

# ---------- canonical huffman ----------
def canon_codes(lens):
    """lens: list of (sym,len). Returns {sym:(code,len)} canonical: sort by (len,sym)."""
    items = sorted(lens, key=lambda t: (t[1], t[0]))
    codes = {}; code = 0; prev = 0
    for s, l in items:
        code <<= (l - prev); prev = l
        codes[s] = (code, l); code += 1
    return codes

class BitReader:
    def __init__(self, data): self.d = data; self.p = 0; self.n = len(data) * 8
    def bit(self):
        if self.p >= self.n: raise SarcError('E_PAYLOAD', 'bitstream end')
        v = (self.d[self.p >> 3] >> (7 - (self.p & 7))) & 1; self.p += 1; return v

def read_tree(buf, o):
    (cnt,) = struct.unpack_from('>H', buf, o); o += 2
    ent = []
    for _ in range(cnt):
        s, l = struct.unpack_from('>HB', buf, o); o += 3; ent.append((s, l))
    return ent, o

def decoder_table(ent):
    if len(ent) == 1:       # degenerate alphabet: single symbol, 1-bit code
        return {(0, 1): ent[0][0]}
    return {(c, l): s for s, (c, l) in canon_codes(ent).items()}

def dec_sym(br, tab, maxlen=32):
    c = 0
    for l in range(1, maxlen + 1):
        c = (c << 1) | br.bit()
        if (c, l) in tab: return tab[(c, l)]
    raise SarcError('E_PAYLOAD', 'bad code')

def decode_block(data, raw_len, out, dist_mode='slot'):
    """Huffman+LZ block. out: bytearray (file history) - appended in place."""
    o = 0
    lt, o = read_tree(data, o); mt, o = read_tree(data, o); dt, o = read_tree(data, o)
    ntok, nbytes = struct.unpack_from('>II', data, o); o += 8
    body = data[o:o + nbytes]
    br = BitReader(body)
    L = decoder_table(lt) if lt else {}
    M = decoder_table(mt) if mt else {}
    D = decoder_table(dt) if dt else {}
    start = len(out)
    for _ in range(ntok):
        s = dec_sym(br, L)
        if s < 256: out.append(s)
        else:
            ln = dec_sym(br, M) + 3
            ds = dec_sym(br, D)
            if ds == 0: dist = 1
            else:
                extra = ds - 1
                v = 1
                for _ in range(extra): v = (v << 1) | br.bit()
                dist = v + 1
            if dist > len(out): raise SarcError('E_PAYLOAD', 'dist')
            for _ in range(ln): out.append(out[-dist])
    if len(out) - start != raw_len: raise SarcError('E_PAYLOAD', 'raw_len mismatch')
