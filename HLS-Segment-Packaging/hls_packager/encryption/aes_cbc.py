"""零依赖 AES-128 + CBC 模式 + PKCS#7 填充实现。

设计目标
--------
HLS 的 ``METHOD=AES-128``（RFC 8216 §4.3.2.4）要求对每个 MPEG-TS 分片做
**AES-128-CBC**：密钥固定 128 位（16 字节），块大小 16 字节；若分片长度
不是块大小的整数倍，按 PKCS#7（RFC 5652 §6.3）在末尾补齐。本模块仅使用
Python 标准库完整实现该密码学原语，不引入任何第三方依赖，与项目
"从零实现"的工程约定保持一致。

模块分三层，自底向上单向依赖：

1. **AES-128 块密码**（FIPS-197）：:func:`aes128_encrypt_block` /
   :func:`aes128_decrypt_block`，一次处理 16 字节；
2. **CBC 模式**（NIST SP 800-38A §6.2）：:class:`CBCEncryptor` /
   :class:`CBCDecryptor` 提供流式 ``update``/``finish`` 接口，
   另提供一次性便捷函数 :func:`cbc_encrypt` / :func:`cbc_decrypt`；
3. **PKCS#7 填充**（RFC 5652）：:func:`pkcs7_pad` / :func:`pkcs7_unpad`，
   去填充时做完整合法性校验，拒绝任何畸形填充。

正确性锚点
----------
``tests/test_encryption_aes.py`` 内置 FIPS-197 附录 B 与 SP 800-38A
F.2.1/F.2.2 的官方已知答案向量（KAT），并在装有 openssl 时做
``openssl enc -aes-128-cbc`` 交叉验证，保证与权威实现字节级一致。

安全说明
--------
* 本模块只承担 HLS 场景下的对称解密兼容义务，密钥来源由上层
  :mod:`hls_packager.encryption.config` 负责；
* CBC 不提供完整性保护（RFC 8216 的 AES-128 方法本身就没有 MAC），
  去填充校验只能发现意外损坏，不能防御主动篡改，这是协议固有属性；
* IV 不可重复用于同一密钥下的不同会话：上层默认每次打包随机生成。
"""

from __future__ import annotations

import os

from ..errors import EncryptionError

#: AES 块大小（字节），AES-128/192/256 均为 128 位块
BLOCK_SIZE = 16
#: AES-128 密钥长度（字节）
KEY_SIZE = 16
#: AES-128 轮数
_ROUNDS = 10
#: PKCS#7 允许的最大填充值（即块大小）
_MAX_PADDING = BLOCK_SIZE

# ---------------------------------------------------------------------------
# GF(2^8) 算术（FIPS-197 §4.2）：S 盒求逆元与 MixColumns 都依赖它，
# 因此先于 S 盒定义。约减模多项式 m(x)=x^8+x^4+x^3+x+1（0x11B）。
# ---------------------------------------------------------------------------
def _xtime(value: int) -> int:
    """GF(2^8) 上乘以 x（即乘 2）：左移一位，溢出则约减模多项式 0x11B。"""
    value <<= 1
    if value & 0x100:
        value ^= 0x11B
    return value & 0xFF


def _gmul(a: int, b: int) -> int:
    """GF(2^8) 乘法，用于 S 盒逆元、MixColumns 的 2/3 与逆变换的 9/11/13/14。"""
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        b >>= 1
        a = _xtime(a)
    return result


# ---------------------------------------------------------------------------
# FIPS-197 §5.1.1 S 盒：GF(2^8) 乘法逆元 + 仿射变换，程序化生成两张表，
# 避免手抄 256 项常量表引入笔误；逆 S 盒由 S 盒置换关系直接求逆。
# ---------------------------------------------------------------------------
def _affine_transform(byte: int) -> int:
    """S 盒仿射变换：b'i = bi⊕b(i+4)⊕b(i+5)⊕b(i+6)⊕b(i+7)⊕ci，c=0x63。"""
    result = 0
    for i in range(8):
        bit = (
            ((byte >> i) & 1)
            ^ ((byte >> ((i + 4) % 8)) & 1)
            ^ ((byte >> ((i + 5) % 8)) & 1)
            ^ ((byte >> ((i + 6) % 8)) & 1)
            ^ ((byte >> ((i + 7) % 8)) & 1)
            ^ ((0x63 >> i) & 1)
        )
        result |= bit << i
    return result


def _build_sboxes() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """构造 (S 盒, 逆 S 盒)：先暴力求 GF(2^8) 逆元，再做仿射变换。"""
    multiplicative_inverse = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if _gmul(a, b) == 1:
                multiplicative_inverse[a] = b
                break
    sbox = tuple(_affine_transform(multiplicative_inverse[a]) for a in range(256))
    inv_sbox = [0] * 256
    for index, value in enumerate(sbox):
        inv_sbox[value] = index
    return sbox, tuple(inv_sbox)


_SBOX, _INV_SBOX = _build_sboxes()

#: FIPS-197 §5.2 轮常数 Rcon（AES-128 只需前 10 项）
_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)


def _key_expansion(key: bytes) -> list[list[int]]:
    """FIPS-197 §5.2 密钥扩展：16 字节种子密钥展开为 11 组轮密钥（每组 16 字节）。"""
    if len(key) != KEY_SIZE:
        raise EncryptionError(
            f"AES-128 密钥必须恰好为 {KEY_SIZE} 字节，实际为 {len(key)} 字节"
        )
    # 以 4 字节字为单位，共 Nb*(Nr+1) = 4*11 = 44 个字
    words = [list(key[i:i + 4]) for i in range(0, KEY_SIZE, 4)]
    for round_index in range(_ROUNDS):
        temp = list(words[-1])
        # RotWord + SubWord + Rcon
        temp = temp[1:] + temp[:1]
        temp = [_SBOX[b] for b in temp]
        temp[0] ^= _RCON[round_index]
        words.append([w ^ t for w, t in zip(words[-4], temp)])
        for _ in range(3):
            words.append([w ^ t for w, t in zip(words[-4], words[-1])])
    # 每 4 个字拼成一组 16 字节轮密钥
    return [sum(words[base:base + 4], []) for base in range(0, 44, 4)]


def _add_round_key(state: list[int], round_key: list[int]) -> None:
    for i in range(BLOCK_SIZE):
        state[i] ^= round_key[i]


def _sub_bytes(state: list[int]) -> None:
    for i in range(BLOCK_SIZE):
        state[i] = _SBOX[state[i]]


def _inv_sub_bytes(state: list[int]) -> None:
    for i in range(BLOCK_SIZE):
        state[i] = _INV_SBOX[state[i]]


def _shift_rows(state: list[int]) -> None:
    """FIPS-197 §5.1.2：状态按列主序存放，行 r 循环左移 r 个字节。"""
    # state[col*4+row]；逐行重排
    rows = [[state[col * 4 + row] for col in range(4)] for row in range(4)]
    for row in range(4):
        rows[row] = rows[row][row:] + rows[row][:row]
    for col in range(4):
        for row in range(4):
            state[col * 4 + row] = rows[row][col]


def _inv_shift_rows(state: list[int]) -> None:
    """逆 ShiftRows：行 r 循环右移 r 个字节。"""
    rows = [[state[col * 4 + row] for col in range(4)] for row in range(4)]
    for row in range(4):
        rows[row] = rows[row][-row:] + rows[row][:-row] if row else rows[row]
    for col in range(4):
        for row in range(4):
            state[col * 4 + row] = rows[row][col]


def _mix_columns(state: list[int]) -> None:
    """FIPS-197 §5.1.3：逐列乘固定矩阵（列内系数 2,3,1,1）。"""
    for col in range(4):
        base = col * 4
        s0, s1, s2, s3 = state[base:base + 4]
        state[base] = _gmul(s0, 2) ^ _gmul(s1, 3) ^ s2 ^ s3
        state[base + 1] = s0 ^ _gmul(s1, 2) ^ _gmul(s2, 3) ^ s3
        state[base + 2] = s0 ^ s1 ^ _gmul(s2, 2) ^ _gmul(s3, 3)
        state[base + 3] = _gmul(s0, 3) ^ s1 ^ s2 ^ _gmul(s3, 2)


def _inv_mix_columns(state: list[int]) -> None:
    """逆 MixColumns：逐列乘逆矩阵（列内系数 14,11,13,9）。"""
    for col in range(4):
        base = col * 4
        s0, s1, s2, s3 = state[base:base + 4]
        state[base] = _gmul(s0, 14) ^ _gmul(s1, 11) ^ _gmul(s2, 13) ^ _gmul(s3, 9)
        state[base + 1] = _gmul(s0, 9) ^ _gmul(s1, 14) ^ _gmul(s2, 11) ^ _gmul(s3, 13)
        state[base + 2] = _gmul(s0, 13) ^ _gmul(s1, 9) ^ _gmul(s2, 14) ^ _gmul(s3, 11)
        state[base + 3] = _gmul(s0, 11) ^ _gmul(s1, 13) ^ _gmul(s2, 9) ^ _gmul(s3, 14)


def aes128_encrypt_block(block: bytes, key: bytes) -> bytes:
    """加密单个 16 字节明文块（FIPS-197 §5.1）。"""
    if len(block) != BLOCK_SIZE:
        raise EncryptionError(
            f"AES 块输入必须恰好为 {BLOCK_SIZE} 字节，实际为 {len(block)} 字节"
        )
    round_keys = _key_expansion(key)
    state = list(block)
    _add_round_key(state, round_keys[0])
    for rnd in range(1, _ROUNDS):
        _sub_bytes(state)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, round_keys[rnd])
    _sub_bytes(state)
    _shift_rows(state)
    _add_round_key(state, round_keys[_ROUNDS])
    return bytes(state)


def aes128_decrypt_block(block: bytes, key: bytes) -> bytes:
    """解密单个 16 字节密文块（FIPS-197 §5.3，加密轮的严格逆序）。"""
    if len(block) != BLOCK_SIZE:
        raise EncryptionError(
            f"AES 块输入必须恰好为 {BLOCK_SIZE} 字节，实际为 {len(block)} 字节"
        )
    round_keys = _key_expansion(key)
    state = list(block)
    _add_round_key(state, round_keys[_ROUNDS])
    for rnd in range(_ROUNDS - 1, 0, -1):
        _inv_shift_rows(state)
        _inv_sub_bytes(state)
        _add_round_key(state, round_keys[rnd])
        _inv_mix_columns(state)
    _inv_shift_rows(state)
    _inv_sub_bytes(state)
    _add_round_key(state, round_keys[0])
    return bytes(state)


# ---------------------------------------------------------------------------
# PKCS#7 填充（RFC 5652 §6.3）
# ---------------------------------------------------------------------------
def pkcs7_pad(data: bytes) -> bytes:
    """按块大小做 PKCS#7 填充：始终追加 1~16 个值等于填充长度的字节。

    数据恰好块对齐时也追加一整块 0x10——这是 PKCS#7 的无歧义约定，
    保证接收方能仅凭末尾字节判断填充长度。
    """
    if not isinstance(data, (bytes, bytearray)):
        raise EncryptionError("待填充数据必须是 bytes/bytearray")
    padding = BLOCK_SIZE - (len(data) % BLOCK_SIZE)
    return bytes(data) + bytes([padding]) * padding


def pkcs7_unpad(data: bytes) -> bytes:
    """校验并移除 PKCS#7 填充；任何畸形填充都抛 :class:`EncryptionError`。

    严格校验三项：长度为块整数倍、末尾填充值在 1~16 内且不超过数据总长、
    填充区内每个字节都等于该长度。任一项失败即判定密文损坏或密钥/IV 错误。
    """
    if not data:
        raise EncryptionError("无法对空数据做 PKCS#7 去填充")
    if len(data) % BLOCK_SIZE != 0:
        raise EncryptionError(
            f"数据长度 {len(data)} 不是块大小 {BLOCK_SIZE} 的整数倍，"
            "密文可能已损坏"
        )
    padding = data[-1]
    if not 1 <= padding <= _MAX_PADDING:
        raise EncryptionError(
            f"非法 PKCS#7 填充值 {padding}（合法范围 1~{_MAX_PADDING}），"
            "密文可能已损坏或密钥/IV 不匹配"
        )
    if padding > len(data):
        raise EncryptionError("填充长度超过数据总长，密文可能已损坏")
    if any(b != padding for b in data[-padding:]):
        raise EncryptionError(
            "PKCS#7 填充区字节不一致，密文可能已损坏或密钥/IV 不匹配"
        )
    return bytes(data[:-padding])


def _xor_blocks(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))


# ---------------------------------------------------------------------------
# CBC 流式加解密（SP 800-38A §6.2）
# ---------------------------------------------------------------------------
class CBCEncryptor:
    """CBC 加密的流式状态机：``update`` 喂任意长度，``finish`` 补齐收尾。

    CBC 链式依赖：每个明文块先与前一密文块（首块用 IV）异或，再送块密码。
    内部缓存不足一块的尾部，因此调用方可以按任意分块（如 64 KiB）喂数据，
    避免把整个分片读进内存。
    """

    def __init__(self, key: bytes, iv: bytes):
        if len(iv) != BLOCK_SIZE:
            raise EncryptionError(
                f"CBC 的 IV 必须恰好为 {BLOCK_SIZE} 字节，实际为 {len(iv)} 字节"
            )
        self._key = bytes(key)
        self._previous = bytes(iv)
        self._buffer = bytearray()
        self._finished = False

    def update(self, data: bytes) -> bytes:
        """摄入数据，返回所有已凑齐块的密文（可能为空）。"""
        if self._finished:
            raise EncryptionError("加密器已 finish，不能再追加数据")
        self._buffer.extend(data)
        ready = len(self._buffer) - len(self._buffer) % BLOCK_SIZE
        chunks: list[bytes] = []
        for offset in range(0, ready, BLOCK_SIZE):
            block = bytes(self._buffer[offset:offset + BLOCK_SIZE])
            encrypted = aes128_encrypt_block(
                _xor_blocks(block, self._previous), self._key
            )
            chunks.append(encrypted)
            self._previous = encrypted
        del self._buffer[:ready]
        return b"".join(chunks)

    def finish(self) -> bytes:
        """追加 PKCS#7 填充并加密最后一块（或两块），返回剩余密文。

        填充可能产生 1~2 个块（缓存恰为 15 字节时补 1 字节得 1 块，
        缓存为空时补整块 0x10 也是 1 块，其余情况最多 2 块），
        逐块走标准链式加密，不经过已收尾的 :meth:`update`。
        """
        if self._finished:
            raise EncryptionError("加密器已 finish，不能重复收尾")
        self._finished = True
        tail = pkcs7_pad(bytes(self._buffer))
        self._buffer.clear()
        chunks: list[bytes] = []
        for offset in range(0, len(tail), BLOCK_SIZE):
            block = tail[offset:offset + BLOCK_SIZE]
            encrypted = aes128_encrypt_block(
                _xor_blocks(block, self._previous), self._key
            )
            chunks.append(encrypted)
            self._previous = encrypted
        return b"".join(chunks)


class CBCDecryptor:
    """CBC 解密的流式状态机，与 :class:`CBCEncryptor` 对称。

    解密链：明文块 = ``AES_decrypt(密文块) XOR 前一密文块``。
    需要缓存最后一个密文块直到 ``finish``——PKCS#7 填充只能在全部
    密文收齐后校验移除。
    """

    def __init__(self, key: bytes, iv: bytes):
        if len(iv) != BLOCK_SIZE:
            raise EncryptionError(
                f"CBC 的 IV 必须恰好为 {BLOCK_SIZE} 字节，实际为 {len(iv)} 字节"
            )
        self._key = bytes(key)
        self._previous = bytes(iv)
        self._buffer = bytearray()
        self._finished = False

    def update(self, data: bytes) -> bytes:
        """摄入密文，返回除最后一个待校验块之外的明文（可能为空）。"""
        if self._finished:
            raise EncryptionError("解密器已 finish，不能再追加数据")
        self._buffer.extend(data)
        # 始终保留最后一个块：填充在末块里，finish 才能处理
        ready = len(self._buffer) - BLOCK_SIZE
        ready -= ready % BLOCK_SIZE
        if ready <= 0:
            return b""
        chunks: list[bytes] = []
        for offset in range(0, ready, BLOCK_SIZE):
            block = bytes(self._buffer[offset:offset + BLOCK_SIZE])
            plain = _xor_blocks(
                aes128_decrypt_block(block, self._key), self._previous
            )
            chunks.append(plain)
            self._previous = block
        del self._buffer[:ready]
        return b"".join(chunks)

    def finish(self) -> bytes:
        """处理末块、校验并移除 PKCS#7 填充，返回剩余明文。"""
        if self._finished:
            raise EncryptionError("解密器已 finish，不能重复收尾")
        self._finished = True
        if len(self._buffer) != BLOCK_SIZE:
            raise EncryptionError(
                f"末段密文长度为 {len(self._buffer)} 字节，"
                f"不是完整的 {BLOCK_SIZE} 字节块，密文可能已截断"
            )
        block = bytes(self._buffer)
        self._buffer.clear()
        plain = _xor_blocks(aes128_decrypt_block(block, self._key), self._previous)
        return pkcs7_unpad(plain)


def cbc_encrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """一次性 CBC 加密（含 PKCS#7 填充），适合小数据与测试。"""
    encryptor = CBCEncryptor(key, iv)
    return encryptor.update(data) + encryptor.finish()


def cbc_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """一次性 CBC 解密（校验并移除 PKCS#7 填充），适合小数据与测试。"""
    decryptor = CBCDecryptor(key, iv)
    return decryptor.update(ciphertext) + decryptor.finish()


def random_iv() -> bytes:
    """生成密码学随机的 16 字节 IV（``os.urandom`` 走系统 CSPRNG）。"""
    return os.urandom(BLOCK_SIZE)
