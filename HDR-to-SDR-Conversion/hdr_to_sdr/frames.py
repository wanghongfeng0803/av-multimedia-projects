"""逐像素转换引擎：解码后原始帧 → SDR 原始帧。

两条输入路径（由 ffprobe 的像素族自动选择，即「位深和色彩空间自适应」）：

* **YUV 路径**（yuv444p16le，所有 YUV 源经 ffmpeg 上采样到 444）：
  码值 → 归一化 YUV → 逆量化到非线性 R'G'B' → EOTF → 线性光
  → 色调映射 → BT.2020→BT.709 色域映射 → BT.709 OETF → 重编码 YUV444。
* **RGB 路径**（gbrpf32le，源为 RGB 像素族时）：直接拿到 0..1 浮点
  非线性 RGB，跳过逆量化，其余流程相同；输出 gbrpf32le 由编码器侧
  ffmpeg 自行转成目标 YUV 格式。

数学模块只提供纯函数，本类负责缓冲解析、LUT 持有与分带并行
（multiprocessing 按水平条带切分；本算法不依赖空间邻域，条带间无上下文，
切分不损失精度）。
"""

from __future__ import annotations

import array
import multiprocessing as mp

from .colorscience import (
    RANGE_LIMITED,
    BT709_KR,
    BT709_KG,
    BT709_KB,
    BT2020_KR,
    BT2020_KB,
    range_constants,
    yuv_to_rgb_offsets,
    rgb_to_yuv_offsets,
    gamut_map_pixel,
)
from .tonemap import (
    DEFAULT_TONE_METHOD,
    resolve_tone_method,
    tone_map_pixel,
)
from .transfer import (
    TRANSFER_PQ,
    TRANSFER_HLG,
    build_eotf_lut,
    build_oetf_lut,
    lerp_lut,
    hlg_display_linear,
)

__all__ = ["FrameProcessor", "DEFAULT_OETF_LUT_SIZE"]

DEFAULT_OETF_LUT_SIZE = 4096
# PQ 元数据缺失时采用的名义峰值（业界常见的 HDR10 母版峰值）
DEFAULT_PQ_PEAK_NITS = 1000.0


class FrameProcessor:
    """持有 LUT 与量化系数的帧处理器（对单帧而言是无状态的纯变换）。"""

    def __init__(
        self,
        transfer: str,
        src_bits: int,
        pixel_range: str,
        is_rgb: bool,
        tone_method: str = DEFAULT_TONE_METHOD,
        peak_nits: float | None = None,
        dst_bits: int = 8,
        dst_range: str = RANGE_LIMITED,
        oetf_lut_size: int = DEFAULT_OETF_LUT_SIZE,
    ):
        if transfer not in (TRANSFER_PQ, TRANSFER_HLG):
            raise ValueError(f"未知传递函数: {transfer}")
        self.transfer = transfer
        self.src_bits = src_bits
        self.pixel_range = pixel_range
        self.is_rgb = is_rgb
        self.tone_method = resolve_tone_method(tone_method)
        self.dst_bits = dst_bits
        self.dst_range = dst_range
        self.bytes_per_sample = 4 if is_rgb else 2
        self.planes = 3  # 两种中间格式都是 444 三平面

        # EOTF LUT 按源码深建表（索引即码值，免插值）；OETF LUT 固定网格+插值
        self._eotf_lut = build_eotf_lut(transfer, src_bits)
        self._oetf_lut = build_oetf_lut(oetf_lut_size)

        # PQ 峰值：元数据有 MaxMDL 用元数据，否则默认 1000nit，归一到 10000nit 域
        if transfer == TRANSFER_PQ:
            peak = (peak_nits or DEFAULT_PQ_PEAK_NITS) / 10000.0
            self.peak = min(max(peak, 1e-4), 1.0)
        else:
            # HLG 经 OOTF 后显示域 1.0 即标称峰值
            self.peak = 1.0

        self._init_quantization()

    # ---- 初始化：量化/逆量化系数 ------------------------------------------
    def _init_quantization(self) -> None:
        if self.is_rgb:
            return  # RGB 路径不涉及 YUV 量化
        # 源：BT.2020 系数逆量化
        (self._src_y_scale, self._src_rv, self._src_gu,
         self._src_gv, self._src_bu) = yuv_to_rgb_offsets(
            BT2020_KR, BT2020_KB, self.pixel_range, self.src_bits
        )
        src_off, _, src_cmin, src_cmax = range_constants(
            self.pixel_range, self.src_bits
        )
        self._src_y_off = float(src_off)
        self._src_c_mid = (src_cmin + src_cmax) / 2.0
        self._src_peak = float((1 << self.src_bits) - 1)
        # 色差展宽（peak/c_span）已包含在 rv/gu/gv/bu 系数内，
        # 像素循环里 un/vn 只做 (code-mid)/peak 的裸归一化，不能再乘一次

        # 目标：BT.709 系数量化
        (self._dst_y_gain, self._dst_c_gain,
         self._dst_kr, self._dst_kg, self._dst_kb,
         self._dst_y_off, self._dst_c_mid) = rgb_to_yuv_offsets(
            BT709_KR, BT709_KB, self.dst_range, self.dst_bits
        )
        dst_off, dst_ymax, dst_cmin, dst_cmax = range_constants(
            self.dst_range, self.dst_bits
        )
        self._dst_y_lo, self._dst_y_hi = dst_off, dst_ymax
        self._dst_c_lo, self._dst_c_hi = dst_cmin, dst_cmax

    # ---- 对外主入口 --------------------------------------------------------
    def process_frame(self, frame: bytes, width: int, height: int) -> bytes:
        """处理一整帧原始字节，返回目标格式的一帧字节。"""
        if self.is_rgb:
            return self._process_rgb(frame, width, height)
        return self._process_yuv(frame, width, height)

    def process_bands(
        self,
        frame: bytes,
        width: int,
        height: int,
        workers: int = 1,
        band_height: int = 64,
    ) -> bytes:
        """按水平条带处理一帧；workers>1 时用进程池并行。

        本算法不依赖空间邻域，条带间无上下文依赖，切分不损失精度。
        进程池用 spawn 上下文（Windows 默认），处理器实例随任务可 pickle。
        """
        if workers <= 1:
            return self.process_frame(frame, width, height)

        bps = self.bytes_per_sample
        plane_bytes = width * height * bps
        band_rows = list(range(0, height, band_height))
        tasks = []
        for row0 in band_rows:
            row1 = min(row0 + band_height, height)
            chunk = self._slice_band(frame, width, row0, row1, plane_bytes, bps)
            tasks.append((chunk, width, row1 - row0))

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(tasks))) as pool:
            band_outs = pool.starmap(
                _band_worker, [(self, *t) for t in tasks]
            )

        # 每个条带输出按 [Y带, U带, V带] 三平面排列，直接 join 会打乱平面布局；
        # 这里先把各条带拆回三个平面，再按平面分别拼接，还原整帧 Y/U/V 顺序。
        plane_chunks: list[list[bytes]] = [[], [], []]
        for band, (_, _, band_h) in zip(band_outs, tasks):
            band_plane_size = width * band_h * bps
            for p in range(3):
                plane_chunks[p].append(
                    band[p * band_plane_size:(p + 1) * band_plane_size]
                )
        return b"".join(
            chunk for plane in plane_chunks for chunk in plane
        )

    @staticmethod
    def _slice_band(
        frame: bytes,
        width: int,
        row0: int,
        row1: int,
        plane_bytes: int,
        bps: int,
    ) -> bytes:
        """从三平面帧中切出 [row0,row1) 行的三个平面并按原顺序拼接。"""
        start = row0 * width * bps
        count = (row1 - row0) * width * bps
        return b"".join(
            frame[p * plane_bytes + start: p * plane_bytes + start + count]
            for p in range(3)
        )

    # ---- YUV 路径 ----------------------------------------------------------
    def _process_yuv(self, frame: bytes, width: int, height: int) -> bytes:
        samples = array.array("H")
        samples.frombytes(frame)
        plane = width * height
        y_plane = samples[0:plane]
        u_plane = samples[plane:2 * plane]
        v_plane = samples[2 * plane:3 * plane]

        out_y = array.array("H", [0]) * plane
        out_u = array.array("H", [0]) * plane
        out_v = array.array("H", [0]) * plane

        oetf = self._oetf_lut
        for i in range(plane):
            # 逆量化到归一化非线性 R'G'B'（BT.2020）
            # y_scale/rv/... 已含全部展宽，码值只做减偏置/peak 的裸归一化
            yn = (y_plane[i] - self._src_y_off) / self._src_peak
            un = (u_plane[i] - self._src_c_mid) / self._src_peak
            vn = (v_plane[i] - self._src_c_mid) / self._src_peak
            rp = yn * self._src_y_scale + self._src_rv * vn
            gp = yn * self._src_y_scale + self._src_gu * un + self._src_gv * vn
            bp = yn * self._src_y_scale + self._src_bu * un

            r_lin, g_lin, b_lin = self._to_display_linear(rp, gp, bp)
            r_lin, g_lin, b_lin = tone_map_pixel(
                r_lin, g_lin, b_lin, self.tone_method, self.peak
            )
            r709, g709, b709 = gamut_map_pixel(r_lin, g_lin, b_lin)
            vr = lerp_lut(oetf, r709)
            vg = lerp_lut(oetf, g709)
            vb = lerp_lut(oetf, b709)
            vr, vg, vb = self._neutral_guard(vr, vg, vb)
            self._encode_yuv(i, vr, vg, vb, out_y, out_u, out_v)

        return out_y.tobytes() + out_u.tobytes() + out_v.tobytes()

    # 三通道差异小于该阈值视为中性轴（数值取自半个 8bit 码级，约 2e-3）
    _NEUTRAL_EPS = 2.0 / 255.0

    def _neutral_guard(
        self, vr: float, vg: float, vb: float
    ) -> tuple[float, float, float]:
        """中性轴保护：近灰像素强制三通道相等，消除矩阵/OETF 累积的灰场发彩。

        真正有彩色的像素通道 spread 远大于阈值，不受影响；阈值取半个 8bit
        码级，只拦截浮点噪声级别的不平衡。
        """
        if max(vr, vg, vb) - min(vr, vg, vb) <= self._NEUTRAL_EPS:
            # 用 BT.709 亮度加权均值而非算术平均，保持灰场亮度不变
            luma = BT709_KR * vr + BT709_KG * vg + BT709_KB * vb
            return luma, luma, luma
        return vr, vg, vb

    def _encode_yuv(
        self,
        i: int,
        vr: float,
        vg: float,
        vb: float,
        out_y: array.ArrayType,
        out_u: array.ArrayType,
        out_v: array.ArrayType,
    ) -> None:
        """把 BT.709 非线性 RGB 编码为目标码深/范围的 YUV 码值。"""
        luma = self._dst_kr * vr + self._dst_kg * vg + self._dst_kb * vb
        # c_gain 已含完整色度增益（limited 8bit 为 112），直接乘色差，不再乘 2
        y_val = luma * self._dst_y_gain + self._dst_y_off
        u_val = (vb - luma) * self._dst_c_gain + self._dst_c_mid
        v_val = (vr - luma) * self._dst_c_gain + self._dst_c_mid

        out_y[i] = min(max(round(y_val), self._dst_y_lo), self._dst_y_hi)
        out_u[i] = min(max(round(u_val), self._dst_c_lo), self._dst_c_hi)
        out_v[i] = min(max(round(v_val), self._dst_c_lo), self._dst_c_hi)

    # ---- RGB 路径 ----------------------------------------------------------
    def _process_rgb(self, frame: bytes, width: int, height: int) -> bytes:
        floats = array.array("f")
        floats.frombytes(frame)
        # gbrpf32le 平面顺序 G,B,R
        plane = width * height
        g_plane = floats[0:plane]
        b_plane = floats[plane:2 * plane]
        r_plane = floats[2 * plane:3 * plane]

        o_g = array.array("f", [0.0]) * plane
        o_b = array.array("f", [0.0]) * plane
        o_r = array.array("f", [0.0]) * plane
        oetf = self._oetf_lut
        for i in range(plane):
            rp = min(max(r_plane[i], 0.0), 1.0)
            gp = min(max(g_plane[i], 0.0), 1.0)
            bp = min(max(b_plane[i], 0.0), 1.0)
            r_lin, g_lin, b_lin = self._to_display_linear(rp, gp, bp)
            r_lin, g_lin, b_lin = tone_map_pixel(
                r_lin, g_lin, b_lin, self.tone_method, self.peak
            )
            r709, g709, b709 = gamut_map_pixel(r_lin, g_lin, b_lin)
            nr, ng, nb = self._neutral_guard(
                lerp_lut(oetf, r709),
                lerp_lut(oetf, g709),
                lerp_lut(oetf, b709),
            )
            o_r[i], o_g[i], o_b[i] = nr, ng, nb
        # 拼回 GBR 平面顺序
        return o_g.tobytes() + o_b.tobytes() + o_r.tobytes()

    # ---- 公共：非线性信号 → 显示线性光 ------------------------------------
    def _to_display_linear(
        self, rp: float, gp: float, bp: float
    ) -> tuple[float, float, float]:
        eotf = self._eotf_lut
        if self.transfer == TRANSFER_PQ:
            # PQ EOTF 单通道独立，信号码值化后查 LUT
            return (
                eotf[self._signal_to_code(rp)],
                eotf[self._signal_to_code(gp)],
                eotf[self._signal_to_code(bp)],
            )
        # HLG：先逆 OETF 得场景线性，再过 OOTF（含亮度耦合，无法 1D 化）
        r_s = eotf[self._signal_to_code(rp)]
        g_s = eotf[self._signal_to_code(gp)]
        b_s = eotf[self._signal_to_code(bp)]
        return hlg_display_linear(r_s, g_s, b_s)

    def _signal_to_code(self, signal: float) -> int:
        """归一化非线性信号 [0,1] 夹取并映射到源码深码值，用于 EOTF 查表。"""
        signal = min(max(signal, 0.0), 1.0)
        return int(signal * ((1 << self.src_bits) - 1) + 0.5)


def _band_worker(
    processor: "FrameProcessor", chunk: bytes, width: int, band_height: int
) -> bytes:
    """进程池入口：条带本身就是缩小版的一帧，直接走整帧处理。"""
    return processor.process_frame(chunk, width, band_height)
