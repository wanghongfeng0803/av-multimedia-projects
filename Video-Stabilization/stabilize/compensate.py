"""补偿与裁剪：用 smooth JSON 驱动 ffmpeg 完成实际像素变换。

设计要点：

- **不在 Python 里搬像素**：逐帧旋转/平移全部交给 ffmpeg 的
  ``rotate`` 与 ``crop`` 滤镜，参数通过 ``sendcmd`` 命令文件按
  帧时间戳下发；Python 只生成文本命令与参数，零像素运算。
- **变换链固定为 sendcmd → rotate → crop**：先绕中心反向旋转，
  再平移。``rotate`` 输出尺寸不变、越界填黑；``crop`` 保持原
  宽高、只移动取样原点，配合 ``max/min`` 表达式钳制永不越界。
- **黑边三种处理模式**（``border_mode``，补偿期参数，不进 JSON，
  因此同一份 smooth JSON 可驱动三种模式、输出同尺寸同帧率，
  结果可直接逐帧对比）：

  - ``fixed`` 固定裁剪（默认，旧行为）：补偿输出带黑边的原画幅
    视频，再由独立 crop 子步骤按统一 inset 裁掉黑边缩回原尺寸；
  - ``dynamic`` 动态裁剪：逐帧只裁该帧实际需要的量（逐帧窗口
    经时域平滑防「呼吸」），帧内直接缩回原尺寸，输出即成品；
  - ``mirror`` 镜像填充：输入先按所需边距做镜像扩边，补偿取样
    落在扩展区内，全程无黑边、不丢画面，输出即成品。

- **裁剪是独立子步骤**（仅 fixed 模式需要）：同样的 smooth JSON
  可被反复复用，``crop`` 子命令读取其中 ``crop.inset`` 做最终
  取景；不重新估计、不重新平滑。
- 音轨：补偿阶段 ``-c:a copy`` 直通；裁剪阶段流重编码，音轨
  同样直通（视频必须重编码）。

角度符号：smooth JSON 的 angle 是 ref→cur 顺时针角，补偿取负；
ffmpeg ``rotate`` 正值为屏幕顺时针，故下发 ``-angle``。
"""

import math
import os
import subprocess

from .pipeline import SMOOTH_FORMAT, PipelineError, _find, fps_to_float
from .smoothing import DEFAULT_RADIUS, dynamic_insets

__all__ = [
    "BORDER_MODES",
    "build_sendcmd",
    "build_sendcmd_dynamic",
    "build_sendcmd_mirror",
    "mirror_margin",
    "compensate_video",
    "crop_video",
    "DEFAULT_CRF",
]

#: 补偿输出的默认 x264 质量（与工具包其他模块一致走质量档）
DEFAULT_CRF = 18

#: 黑边处理模式：fixed 固定裁剪（默认）/ dynamic 动态裁剪 / mirror 镜像填充
BORDER_MODES = ("fixed", "dynamic", "mirror")


def _frame_pts(doc, index):
    """第 index 帧的时间戳（秒），按 CFR 帧率从 fps 字段推导。"""
    fps = fps_to_float(doc.get("fps") or "25")
    return index / fps


def build_sendcmd(doc, cmd_path):
    """由 smooth JSON 生成 ffmpeg sendcmd 命令文件。

    目标滤镜带实例标签（``@st`` 旋转、``@cp`` 裁剪），每帧在其
    PTS 处同时更新三个参数：旋转角（弧度）与裁剪原点 x/y。

    平移实现：rotate 之后的画面与画布同尺寸，直接平移会露黑边，
    因此 crop 窗口取「原尺寸 − 2·安全边距」的固定小框（边距取自
    JSON 的推荐裁剪 inset，保证任何一帧的取样框都在画面内），
    原点在 ``[inset−补偿, inset+补偿]`` 区间移动，再用 pad 补回
    原尺寸。最终 crop 子步骤会把这条安全边一并裁掉。
    """
    if doc.get("format") != SMOOTH_FORMAT:
        raise PipelineError(
            f"需要 {SMOOTH_FORMAT} 参数文件，收到 {doc.get('format')!r}"
        )
    w, h = doc["width"], doc["height"]
    inset = float(doc["crop"]["inset"])
    cw, ch = w - 2 * int(round(inset)), h - 2 * int(round(inset))
    if cw < 2 or ch < 2:
        raise PipelineError(
            f"推荐裁剪 inset={inset} 已超出画幅，无法构造补偿取样框"
        )
    lines = [
        "# stabilize sendcmd（自动生成，帧号 → 旋转/平移参数）",
        f"# 取样框 {cw}x{ch}，原点活动区间 [{inset:.3f}±补偿]",
    ]
    for f in doc["frames"]:
        comp = f["compensation"]
        pts = _frame_pts(doc, f["index"])
        angle_rad = -math.radians(comp["angle"])
        # 原点 = 居中位置 − 补偿位移；钳到合法 [0, w-cw]/[0, h-ch]
        ox = inset - comp["dx"]
        oy = inset - comp["dy"]
        ox_expr = f"min(max({ox:.6f}\\,0)\\,{w - cw})"
        oy_expr = f"min(max({oy:.6f}\\,0)\\,{h - ch})"
        lines.append(
            f"{pts:.6f} [enter] "
            f"rotate@st angle {angle_rad:.8f}, "
            f"crop@cp x {ox_expr}, "
            f"crop@cp y {oy_expr};"
        )
    with open(cmd_path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(lines) + "\n")
    return cmd_path


def _check_smooth_doc(doc):
    if doc.get("format") != SMOOTH_FORMAT:
        raise PipelineError(
            f"需要 {SMOOTH_FORMAT} 参数文件，收到 {doc.get('format')!r}"
        )


def build_sendcmd_dynamic(doc, cmd_path, insets):
    """动态裁剪模式的 sendcmd：逐帧下发旋转角与裁剪窗口四参数。

    每帧窗口 ``(w−2iₜ)×(h−2iₜ)`` 只裁该帧实际需要的量（``insets``
    由 :func:`stabilize.smoothing.dynamic_insets` 时域平滑，抑制
    逐帧缩放比突变造成的「呼吸」），原点 ``(iₜ−补偿)`` 钳在合法
    区间；窗口尺寸逐帧变化，下游 ``scale=eval=frame`` 把每帧缩回
    原画幅，输出即成品（无黑边、无需 crop 子步骤）。
    """
    _check_smooth_doc(doc)
    w, h = doc["width"], doc["height"]
    frames = doc["frames"]
    if len(insets) != len(frames):
        raise PipelineError(
            f"逐帧 inset 数量（{len(insets)}）与帧数（{len(frames)}）不符"
        )
    lines = [
        "# stabilize sendcmd（动态裁剪，帧号 → 旋转角 + 逐帧窗口）",
        f"# 画幅 {w}x{h}，窗口逐帧变化，下游 scale 缩回原尺寸",
    ]
    for f, ins in zip(frames, insets):
        ins = int(ins)
        cw, ch = w - 2 * ins, h - 2 * ins
        if cw < 2 or ch < 2:
            raise PipelineError(
                f"第 {f['index']} 帧动态 inset={ins} 超出画幅，"
                f"剩余 {cw}x{ch}"
            )
        comp = f["compensation"]
        pts = _frame_pts(doc, f["index"])
        angle_rad = -math.radians(comp["angle"])
        ox = ins - comp["dx"]
        oy = ins - comp["dy"]
        ox_expr = f"min(max({ox:.6f}\\,0)\\,{w - cw})"
        oy_expr = f"min(max({oy:.6f}\\,0)\\,{h - ch})"
        lines.append(
            f"{pts:.6f} [enter] "
            f"rotate@st angle {angle_rad:.8f}, "
            f"crop@cp w {cw}, "
            f"crop@cp h {ch}, "
            f"crop@cp x {ox_expr}, "
            f"crop@cp y {oy_expr};"
        )
    with open(cmd_path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(lines) + "\n")
    return cmd_path


def mirror_margin(doc):
    """镜像填充所需的四边扩边像素（整数）。

    取样窗口恒为原画幅 w×h，原点在 ``[0, 2m]`` 内移动，要求扩边
    后的画布同时容纳平移峰值与旋转角点：设 ``s=|sinθ|``、
    ``c=1−cosθ``（θ 为全片最大补偿角），扩展画布的旋转伸出量为
    ``rotation_inset(w,h,θ) + m·(s+c)``，解
    ``m ≥ (平移峰值 + 旋转伸出 + 安全余量) / (1 − s − c)``。
    分子即 smooth JSON 的 ``crop.raw_inset``（上限钳制前的真实
    需求）。角度过大（s+c ≥ 0.5，约 30° 以上）时扩边发散，拒绝。
    """
    _check_smooth_doc(doc)
    crop = doc["crop"]
    base = float(crop.get("raw_inset") or crop["inset"])
    max_angle = abs(float(crop.get("max_angle", 0.0)))
    theta = math.radians(max_angle)
    k = abs(math.sin(theta)) + (1.0 - math.cos(theta))
    if k >= 0.5:
        raise PipelineError(
            f"最大补偿角 {max_angle}° 过大，镜像扩边发散，"
            f"请改用 fixed/dynamic 模式或开启补偿限幅"
        )
    return int(math.ceil(base / (1.0 - k))) + 1


def build_sendcmd_mirror(doc, cmd_path, margin):
    """镜像填充模式的 sendcmd：窗口恒为原画幅，原点在扩展画布内移动。

    输入先经 ``fillborders`` 四边镜像扩边 ``margin`` 像素，旋转与
    平移的取样全部落在镜像扩展区内，全程无黑边、不丢画面，输出
    即成品。每帧只下发旋转角与窗口原点 ``(margin−补偿)``。
    """
    _check_smooth_doc(doc)
    w, h = doc["width"], doc["height"]
    margin = int(margin)
    if margin < 1:
        raise PipelineError(f"镜像扩边量 {margin} 非法")
    lines = [
        "# stabilize sendcmd（镜像填充，帧号 → 旋转角 + 窗口原点）",
        f"# 画布四边镜像扩边 {margin}px，窗口恒为 {w}x{h}",
    ]
    for f in doc["frames"]:
        comp = f["compensation"]
        pts = _frame_pts(doc, f["index"])
        angle_rad = -math.radians(comp["angle"])
        ox = margin - comp["dx"]
        oy = margin - comp["dy"]
        ox_expr = f"min(max({ox:.6f}\\,0)\\,{2 * margin})"
        oy_expr = f"min(max({oy:.6f}\\,0)\\,{2 * margin})"
        lines.append(
            f"{pts:.6f} [enter] "
            f"rotate@st angle {angle_rad:.8f}, "
            f"crop@cp x {ox_expr}, "
            f"crop@cp y {oy_expr};"
        )
    with open(cmd_path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(lines) + "\n")
    return cmd_path


def _encoder_args(dst, has_audio, src, crf):
    """组装编码段：libx264 + yuv420p，音轨直通。"""
    args = ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if has_audio:
        args += ["-map", "1:a:0?", "-c:a", "copy"]
    args += [str(dst)]
    return args


def _filter_path(path):
    """把命令文件路径转成 ffmpeg 滤镜安全的正斜杠相对路径。

    ffmpeg 滤镜语法中冒号是选项分隔符，Windows 盘符 ``C:`` 的冒号
    转义在各版本行为不一致，反斜杠又会被当转义符。因此命令文件
    一律落在当前工作目录可相对寻址的位置，这里只做反斜杠→正斜杠
    的归一化；调用方负责保证路径可相对化（见 :func:`compensate_video`）。
    """
    return os.fspath(path).replace("\\", "/")


def compensate_video(doc, src, dst, work_dir, crf=DEFAULT_CRF,
                     progress=None, border_mode="fixed"):
    """补偿阶段：sendcmd + rotate + crop 滤镜输出稳定视频。

    ``border_mode`` 决定黑边处理方式（详见模块 docstring）：

    - ``fixed``（默认，旧行为）：输出原画幅含黑边视频，需再接
      :func:`crop_video` 裁边缩回；
    - ``dynamic``：逐帧窗口 + 帧内缩回原尺寸，输出即成品；
    - ``mirror``：镜像扩边取样，输出即成品（无黑边、不丢画面）。

    ``work_dir`` 存放临时 sendcmd 文件（调用方管理生命周期）。
    返回统计 dict。输出帧数/帧率与输入一致，音轨直通。
    """
    if border_mode not in BORDER_MODES:
        raise PipelineError(
            f"未知黑边模式 {border_mode!r}，可选 {BORDER_MODES}"
        )
    ffmpeg = _find("ffmpeg")
    os.makedirs(work_dir, exist_ok=True)
    # sendcmd 文件路径会进 ffmpeg 滤镜字符串，Windows 绝对路径的
    # 盘符冒号无法可靠转义；故在当前工作目录下建一个专用临时目录
    # 承载命令文件，保证滤镜里始终是无盘符的相对路径。
    import tempfile
    cmd_holder = tempfile.mkdtemp(prefix=".stabilize_cmd_", dir=os.getcwd())
    cmd_path = os.path.join(cmd_holder, "stabilize.cmd")

    w, h = doc["width"], doc["height"]
    # 相对当前工作目录的路径，避开 Windows 盘符冒号的滤镜转义问题
    cmd_filter = _filter_path(os.path.relpath(cmd_path))

    if border_mode == "dynamic":
        frames = doc["frames"]
        insets = dynamic_insets(
            w, h,
            [f["compensation"]["dx"] for f in frames],
            [f["compensation"]["dy"] for f in frames],
            [f["compensation"]["angle"] for f in frames],
            radius=int(doc.get("params", {}).get("radius", DEFAULT_RADIUS)),
        )
        build_sendcmd_dynamic(doc, cmd_path, insets)
        # 滤镜初始参数取第 0 帧取值（首条 sendcmd 在 pts=0 生效前的兜底）
        i0 = insets[0]
        cw0, ch0 = w - 2 * i0, h - 2 * i0
        f0 = frames[0]["compensation"]
        ox0 = min(max(i0 - f0["dx"], 0.0), float(w - cw0))
        oy0 = min(max(i0 - f0["dy"], 0.0), float(h - ch0))
        # 逐帧窗口承载平移与裁剪，scale 逐帧重评估缩回原画幅
        vf = (
            f"sendcmd=f={cmd_filter},"
            f"rotate@st=angle=0:fillcolor=black:bilinear=1,"
            f"crop@cp={cw0}:{ch0}:x={ox0:.3f}:y={oy0:.3f},"
            f"scale={w}:{h}:flags=lanczos:eval=frame"
        )
    elif border_mode == "mirror":
        margin = mirror_margin(doc)
        build_sendcmd_mirror(doc, cmd_path, margin)
        f0 = doc["frames"][0]["compensation"]
        ox0 = min(max(margin - f0["dx"], 0.0), float(2 * margin))
        oy0 = min(max(margin - f0["dy"], 0.0), float(2 * margin))
        # 先镜像扩边再补偿：取样永远落在扩展区内，全程无黑边
        vf = (
            f"fillborders={margin}:{margin}:{margin}:{margin}:mode=mirror,"
            f"sendcmd=f={cmd_filter},"
            f"rotate@st=angle=0:fillcolor=black:bilinear=1,"
            f"crop@cp={w}:{h}:x={ox0:.3f}:y={oy0:.3f}"
        )
    else:
        build_sendcmd(doc, cmd_path)
        inset = int(round(float(doc["crop"]["inset"])))
        cw, ch = w - 2 * inset, h - 2 * inset
        # 旋转保持原画幅（越界填黑）→ 小框 crop 承载平移 → pad 回原尺寸。
        # pad 的居中偏移使小框落在原画布中央，边缘黑边交给 crop 子步骤。
        vf = (
            f"sendcmd=f={cmd_filter},"
            f"rotate@st=angle=0:fillcolor=black:bilinear=1,"
            f"crop@cp={cw}:{ch}:x=0:y=0,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    # work_dir 仍保留一份命令文件副本，供人工检查与测试断言
    inspect_path = os.path.join(work_dir, "stabilize.cmd")
    if os.path.abspath(inspect_path) != os.path.abspath(cmd_path):
        with open(cmd_path, "rb") as src_fp, open(inspect_path, "wb") as dst_fp:
            dst_fp.write(src_fp.read())

    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src)]
    if doc.get("has_audio"):
        cmd += ["-i", str(src)]
    cmd += ["-vf", vf]
    cmd += _encoder_args(dst, bool(doc.get("has_audio")), src, crf)

    try:
        proc = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    finally:
        import shutil
        shutil.rmtree(cmd_holder, ignore_errors=True)
    if proc.returncode != 0:
        raise PipelineError(
            f"ffmpeg 补偿失败（退出码 {proc.returncode}），"
            f"可检查命令文件副本 {inspect_path}"
        )
    if progress:
        progress(len(doc["frames"]))
    return {
        "frames": len(doc["frames"]),
        "width": w,
        "height": h,
        "border_mode": border_mode,
        "final": border_mode != "fixed",
        "sendcmd": inspect_path,
        "output": os.fspath(dst),
    }


def crop_video(doc, src, dst, inset=None, crf=DEFAULT_CRF):
    """裁剪阶段：按 smooth JSON 的 inset 裁掉黑边并缩回原分辨率。

    补偿阶段的输出是「原画幅 + 四周 inset 宽黑边」的居中画面，
    本步把中间 ``(w-2·inset)×(h-2·inset)`` 的有效区域取出再
    Lanczos 缩回原宽高。``inset`` 显式给出时覆盖 JSON 推荐值
    （仍做合法性钳制）。帧率不变，音轨直通。
    """
    w, h = doc["width"], doc["height"]
    if inset is None:
        inset = float(doc["crop"]["inset"])
    inset_i = int(round(inset))
    max_inset = (min(w, h) - 2) // 2
    inset_i = max(0, min(inset_i, max_inset))
    cw, ch = w - 2 * inset_i, h - 2 * inset_i
    if cw < 2 or ch < 2:
        raise PipelineError(f"裁剪量 {inset_i} 过大，剩余画幅 {cw}x{ch}")

    ffmpeg = _find("ffmpeg")
    vf = (
        f"crop={cw}:{ch}:"
        f"x=(in_w-{cw})/2:y=(in_h-{ch})/2,"
        f"scale={w}:{h}:flags=lanczos"
    )
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src)]
    if doc.get("has_audio"):
        cmd += ["-i", str(src)]
    cmd += ["-vf", vf]
    cmd += _encoder_args(dst, bool(doc.get("has_audio")), src, crf)

    proc = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise PipelineError(f"ffmpeg 裁剪失败（退出码 {proc.returncode}）")
    return {
        "inset": float(inset_i),
        "crop_width": cw,
        "crop_height": ch,
        "width": w,
        "height": h,
        "output": os.fspath(dst),
    }
