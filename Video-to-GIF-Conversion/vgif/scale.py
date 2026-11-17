"""纯 Python 缩放：RGB 双线性/面积平均（定点可复现）与索引帧最近邻。

缩放严格位于三步管线的**抽帧之后、量化之前**（RGB 域），保证调色板
基于最终尺寸生成。两条 RGB 路径按缩放方向自动选择，避免锯齿：

- :func:`bilinear_resize` 在 **RGB 域**做双线性插值，用于**放大**
  （dst ≥ src）。放大时不存在欠采样，双线性的 2×2 邻域混合即可
  平滑过渡；坐标映射用 1/65536 定点，全整数插值，跨平台逐字节
  可复现（与 transition 项目 resample 的定点口径一致）。
- :func:`area_resize` 在 **RGB 域**做面积平均（box filter），用于
  **缩小**（dst < src）。直接双线性缩小会让每个输出像素只覆盖源图
  2×2 邻域，落在采样点之间的高频细节被直接丢弃而产生锯齿/摩尔纹；
  面积平均把每个输出像素覆盖的源区间内**全部**像素按覆盖长度加权
  平均，等价于先低通再重采样，从根因上抗锯齿。
- :func:`resize_rgb` 是统一入口，按目标与源尺寸逐轴选择算法（宽缩
  小走面积、高放大走双线性，可混用），管线只依赖这一个函数。

另有 :func:`nearest_resize_indexed` 在 **索引域**工作，直接对调色板
下标取最近邻，无需重新量化、零颜色损失且极快，仅供已量化产物改
尺寸时复用（适合像素风 / 图表类素材）；主转换路径不使用它。
"""

from __future__ import annotations

__all__ = [
    "SHIFT",
    "SCALE",
    "round_odd",
    "bilinear_resize",
    "area_resize",
    "resize_rgb",
    "nearest_resize_indexed",
]

#: 定点小数位数（2^16），坐标与权重都放大到该尺度做整数运算。
SHIFT = 16
SCALE = 1 << SHIFT


def round_odd(value: int, minimum: int = 1) -> int:
    """把整数规整到最近的奇数，且不小于 ``minimum``（最小奇数 1）。

    奇数对齐口径：偶数 ``v`` 与相邻两个奇数等距（各差 1），统一
    **向下取奇**（``v-1``），规则确定、不引入随机方向；只有向下会
    跌破下限（v=2 而 minimum≥1 时 v-1=1 仍合法；v=0 等边界）才向上
    取奇。下限本身也会被规整到奇数。
    """
    if minimum < 1:
        minimum = 1
    if minimum % 2 == 0:
        minimum += 1
    v = int(value)
    if v % 2 == 0:
        v -= 1
    if v < minimum:
        v = minimum
        if v % 2 == 0:
            v += 1
    return v


def _round_shift2(x: int) -> int:
    """右移 2*SHIFT 位并半值向上取整。

    双线性的两个权重各为 16 位定点，相乘后权重和是 32 位定点
    （SCALE²），故累计值要右移 32 位而非 16 位，否则结果放大 65536
    倍溢出字节。
    """
    return (x + (1 << (2 * SHIFT - 1))) >> (2 * SHIFT)


def _validate_frame(frame: bytes, src_w: int, src_h: int) -> None:
    if len(frame) != src_w * src_h * 3:
        raise ValueError(
            f"帧长度 {len(frame)} 与 {src_w}x{src_h} rgb24 不符"
        )
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"源尺寸必须为正：{src_w}x{src_h}")


def _validate_dst(dst_w: int, dst_h: int) -> None:
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"目标尺寸必须为正：{dst_w}x{dst_h}")


def bilinear_resize(frame: bytes, src_w: int, src_h: int,
                    dst_w: int, dst_h: int) -> bytes:
    """把一帧 rgb24 字节双线性缩放到 ``dst_w × dst_h``。

    采用「角点对齐」坐标映射：输出像素中心 ``(dx+0.5)/dst_w`` 映回
    源图 ``sx = (dx+0.5)*src_w/dst_w - 0.5``，越界坐标钳到边界。
    权重以 1/65536 定点表示，插值结果四舍五入到整数，恒等尺寸
    （src==dst）时逐字节等于输入。

    适用方向：**放大**。缩小时请用 :func:`area_resize` 抗锯齿，或
    直接调用自动选路的 :func:`resize_rgb`。

    :param frame: 长度 ``src_w*src_h*3`` 的 rgb24 字节。
    :return: 长度 ``dst_w*dst_h*3`` 的 rgb24 字节。
    """
    _validate_frame(frame, src_w, src_h)
    _validate_dst(dst_w, dst_h)
    if dst_w == src_w and dst_h == src_h:
        return bytes(frame)

    out = bytearray(dst_w * dst_h * 3)
    for dy in range(dst_h):
        # 映回源坐标（定点）。
        sy_num = (2 * dy + 1) * src_h - dst_h       # * dst_h 后的分子
        sy_q = sy_num * SCALE // (2 * dst_h)        # 源 y 定点坐标
        y0 = sy_q >> SHIFT
        wy = sy_q & (SCALE - 1)
        if y0 < 0:
            y0, wy = 0, 0
        y1 = y0 + 1
        if y1 >= src_h:
            y1 = y0
            wy = 0
        row0 = y0 * src_w * 3
        row1 = y1 * src_w * 3

        for dx in range(dst_w):
            sx_num = (2 * dx + 1) * src_w - dst_w
            sx_q = sx_num * SCALE // (2 * dst_w)
            x0 = sx_q >> SHIFT
            wx = sx_q & (SCALE - 1)
            if x0 < 0:
                x0, wx = 0, 0
            x1 = x0 + 1
            if x1 >= src_w:
                x1 = x0
                wx = 0

            i00 = row0 + x0 * 3
            i01 = row0 + x1 * 3
            i10 = row1 + x0 * 3
            i11 = row1 + x1 * 3
            # 双线性权重（定点）：w00=(1-wx)(1-wy) 等，和约为 SCALE。
            w00 = (SCALE - wx) * (SCALE - wy)
            w01 = wx * (SCALE - wy)
            w10 = (SCALE - wx) * wy
            w11 = wx * wy
            op = (dy * dst_w + dx) * 3
            for c in range(3):
                v = (
                    frame[i00 + c] * w00
                    + frame[i01 + c] * w01
                    + frame[i10 + c] * w10
                    + frame[i11 + c] * w11
                )
                out[op + c] = _round_shift2(v)
    return bytes(out)


def _axis_spans(src_n: int, dst_n: int) -> tuple:
    """计算一条轴上每个输出单元覆盖的源像素区间与定点覆盖权重。

    角点对齐口径与双线性一致：输出单元 ``d`` 覆盖源轴区间
    ``[d*src/dst, (d+1)*src/dst)``。每个被覆盖的源像素 ``i`` 的
    权重为区间交叠长度（1/65536 定点）；单元内权重和等于该单元在
    源轴上的长度（定点），各单元求和恰好为 ``src_n*SCALE``，无缺口、
    无重叠。

    :return: 两个列表 ``(cells, wsums)``：``cells[d]`` 为
        ``[(src_index, weight_q), ...]``（按源像素顺序），
        ``wsums[d]`` 为该单元的权重和（定点，供归一化）。
    """
    cells = []
    wsums = []
    for d in range(dst_n):
        lo = d * src_n * SCALE
        hi = (d + 1) * src_n * SCALE
        lo_q, hi_q = lo // dst_n, hi // dst_n
        i0 = lo_q // SCALE
        i1 = min(src_n - 1, (hi_q - 1) // SCALE)
        weights = []
        for i in range(i0, i1 + 1):
            cell_lo = i * SCALE
            cell_hi = (i + 1) * SCALE
            w = min(hi_q, cell_hi) - max(lo_q, cell_lo)
            if w > 0:
                weights.append((i, w))
        cells.append(weights)
        wsums.append(hi_q - lo_q)
    return cells, wsums


def area_resize(frame: bytes, src_w: int, src_h: int,
                dst_w: int, dst_h: int) -> bytes:
    """把一帧 rgb24 字节用面积平均（box filter）缩放到 ``dst_w × dst_h``。

    每个输出像素取其在源图上对应矩形区域内全部像素、按面积交叠长度
    加权的平均值。缩小时一个输出像素可覆盖多个源像素，高频细节被
    低通平均而非直接丢弃，因此**抗锯齿**（双线性缩小只碰 2×2 邻域，
    会漏采产生摩尔纹/锯齿）。

    实现为水平、竖直两遍可分离卷积，全部 1/65536 定点整数运算。
    权重和随缩放比变化（缩小后一个单元覆盖多个源像素，放大后不足
    一个），因此不能按固定位数右移，而是按每个输出像素的实际权重
    总和 ``wsum_x*wsum_y`` 做整数归一化（半值向上取整）——纯色在
    任意缩放比下都逐值不变，恒等尺寸时逐字节等于输入。

    :param frame: 长度 ``src_w*src_h*3`` 的 rgb24 字节。
    :return: 长度 ``dst_w*dst_h*3`` 的 rgb24 字节。
    """
    _validate_frame(frame, src_w, src_h)
    _validate_dst(dst_w, dst_h)
    if dst_w == src_w and dst_h == src_h:
        return bytes(frame)

    x_cells, x_wsum = _axis_spans(src_w, dst_w)
    y_cells, y_wsum = _axis_spans(src_h, dst_h)

    # ---- 第一遍：水平。中间帧宽 dst_w、高 src_h，保留定点累计值。----
    mid = [[0] * (dst_w * 3) for _ in range(src_h)]
    for sy in range(src_h):
        row_in = sy * src_w * 3
        row_out = mid[sy]
        for dx, weights in enumerate(x_cells):
            op = dx * 3
            acc = [0, 0, 0]
            for sx, wq in weights:
                ip = row_in + sx * 3
                for c in range(3):
                    acc[c] += frame[ip + c] * wq
            for c in range(3):
                row_out[op + c] = acc[c]

    # ---- 第二遍：竖直，按实际权重总和整数归一化回字节。----
    out = bytearray(dst_w * dst_h * 3)
    for dy, weights in enumerate(y_cells):
        wy_total = y_wsum[dy]
        op_row = dy * dst_w * 3
        for dx in range(dst_w):
            mp = dx * 3
            acc = [0, 0, 0]
            for sy, wq in weights:
                row = mid[sy]
                for c in range(3):
                    acc[c] += row[mp + c] * wq
            total = x_wsum[dx] * wy_total
            half = total // 2
            op = op_row + mp
            for c in range(3):
                v = (acc[c] + half) // total
                out[op + c] = v if v <= 255 else 255
    return bytes(out)


def resize_rgb(frame: bytes, src_w: int, src_h: int,
               dst_w: int, dst_h: int) -> bytes:
    """RGB 域统一缩放入口：缩小走面积平均抗锯齿，放大走双线性。

    选路**逐轴独立**：宽、高各自比较源/目标尺寸，因此「宽缩小、高
    放大」这类混合情形由两次单轴变换级联完成，每一段都使用匹配其
    方向的算法，全程在抽帧之后、量化之前的 RGB 域执行。

    - 两轴都不变：原样返回（逐字节）；
    - 两轴同向且算法相同：一次二维变换完成；
    - 两轴算法不同：先水平后竖直两次单轴级联，中间帧是普通 rgb24
      字节帧（面积/双线性各自出口都已归一回字节），几何为
      ``(dst_w, src_h)``。

    :param frame: 长度 ``src_w*src_h*3`` 的 rgb24 字节。
    :return: 长度 ``dst_w*dst_h*3`` 的 rgb24 字节。
    """
    _validate_frame(frame, src_w, src_h)
    _validate_dst(dst_w, dst_h)
    if dst_w == src_w and dst_h == src_h:
        return bytes(frame)

    shrink_w = dst_w < src_w
    shrink_h = dst_h < src_h
    if shrink_w == shrink_h:
        # 同向：两轴同算法，一次完成。
        if shrink_w:
            return area_resize(frame, src_w, src_h, dst_w, dst_h)
        return bilinear_resize(frame, src_w, src_h, dst_w, dst_h)

    # 混合方向：先水平（src_w → dst_w，高保持 src_h），再竖直。
    if shrink_w:
        mid = area_resize(frame, src_w, src_h, dst_w, src_h)
        return bilinear_resize(mid, dst_w, src_h, dst_w, dst_h)
    mid = bilinear_resize(frame, src_w, src_h, dst_w, src_h)
    return area_resize(mid, dst_w, src_h, dst_w, dst_h)


def nearest_resize_indexed(indices: bytes, src_w: int, src_h: int,
                           dst_w: int, dst_h: int) -> bytes:
    """对调色板索引帧做最近邻缩放（不经过 RGB、不重新量化）。

    索引值是调色板下标而非颜色分量，任何加权混合都会造出不存在的
    下标，因此只做最近邻；角点对齐映射保证恒等尺寸时逐字节不变。

    :param indices: 长度 ``src_w*src_h`` 的索引字节（每像素 1 字节）。
    :return: 长度 ``dst_w*dst_h`` 的索引字节。
    """
    if len(indices) != src_w * src_h:
        raise ValueError(
            f"索引帧长度 {len(indices)} 与 {src_w}x{src_h} 不符"
        )
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"目标尺寸必须为正：{dst_w}x{dst_h}")
    if dst_w == src_w and dst_h == src_h:
        return bytes(indices)

    out = bytearray(dst_w * dst_h)
    for dy in range(dst_h):
        sy_num = (2 * dy + 1) * src_h - dst_h
        sy_q = sy_num // (2 * dst_h)
        y = sy_q if sy_q >= 0 else 0
        if y >= src_h:
            y = src_h - 1
        for dx in range(dst_w):
            sx_num = (2 * dx + 1) * src_w - dst_w
            sx_q = sx_num // (2 * dst_w)
            x = sx_q if sx_q >= 0 else 0
            if x >= src_w:
                x = src_w - 1
            out[dy * dst_w + dx] = indices[y * src_w + x]
    return bytes(out)
