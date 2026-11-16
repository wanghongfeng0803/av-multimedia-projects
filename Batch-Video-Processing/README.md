# Batch-Video-Processing（bvp）

可复核的**目录批量视频转码**工具：单文件失败不中断整体，成功/失败/跳过
分别落结构化清单，清单数字与磁盘产出可逐条对账。仅依赖 Python 标准库，
ffmpeg / ffprobe 通过子进程调用。

## 设计目标与硬约束

1. **单文件失败不中断整体**：任何单文件错误（含核心契约外的陌生异常）
   都转成一条清单条目，工作线程继续处理后续任务。
2. **失败与跳过分别落盘**：全量清单之外，失败、跳过各有独立清单文件。
3. **数字可复核**：清单 `summary` 全部由条目实时计数，满足
   `total == success + failed + skipped`；落盘后立即对账，并提供脱离运行
   上下文的 `verify` 子命令。
4. **稳定错误分类契约**：批量层只依据核心异常的 `ErrorCategory` 分类
   汇总，**不解析异常消息、不匹配类名字符串**。
5. **遍历顺序确定**：手写 `os.scandir` 深度优先，每层按
   `(小写名, 原名)` 排序，不依赖文件系统返回次序；正确处理嵌套目录与
   无法访问的文件。
6. **批量层只复用单文件核心**：`bvp.batch` 不拼任何 ffmpeg 命令、不做
   媒体探测，全部通过 `bvp.core.transcode()` 复用。

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `bvp/contract.py` | 稳定契约：`ErrorCategory` 枚举、跳过分类集合、`classify_error()` |
| `bvp/errors.py` | 核心异常体系，每个异常在类级绑定一个 `category` |
| `bvp/core.py` | 单文件转码核心：探测→规划→编码→复核，失败清理半成品 |
| `bvp/scanner.py` | 确定性目录扫描，不可访问路径上报为 `ScanProblem` |
| `bvp/planning.py` | 输出路径镜像规划与同批冲突裁决（纯字符串计算） |
| `bvp/manifest.py` | 三份清单的数据模型、原子落盘与回读、计数不变量 |
| `bvp/verify.py` | 清单条目与磁盘事实逐条对账 |
| `bvp/batch.py` | 批量编排（只调核心、只按契约分类） |
| `bvp/cli.py` | 命令行入口 `python -m bvp` |

## 错误分类契约（核心 → 批量层）

`ErrorCategory` 是闭合枚举，取值字符串是持久化契约，不可随手改名：

| 分类 | 含义 | 批量口径 |
| --- | --- | --- |
| `binary_missing` | 找不到 ffmpeg/ffprobe | 失败 |
| `input_not_found` | 输入不存在/无法访问 | 失败 |
| `not_a_video` | 非媒体、无法识别、无视频流、0 字节 | **跳过** |
| `output_exists` | 输出已存在且策略为 skip | **跳过** |
| `transcode_failed` | ffmpeg 执行失败/产物复核不过 | 失败 |
| `encoder_unavailable` | 本机不支持所需编码器 | 失败 |
| `internal_error` | 契约外未知异常的兜底分类 | 失败 |

跳过边界由 `contract.SKIP_CATEGORIES = {not_a_video, output_exists}` 唯一
划定。核心异常在**类定义处**固定分类，调用方无法在 raise 时临时篡改；
任何没有 `category` 属性的异常一律归入 `internal_error`，绝不静默吞掉。

## 清单文件

默认写入输出目录（可用 `--manifest-dir` 覆盖）：

- `batch_manifest.json`：全量清单，每个输入恰好一条，按 `index` 排序；
- `batch_failures.json`：失败条目（全量清单的过滤视图，共享同一 summary）；
- `batch_skipped.json`：跳过条目。

写入采用「同目录临时文件 + `os.replace`」原子替换，中断不会留下半截 JSON。
成功条目记录 `output_size` 作为复核锚点。

## 用法

```bash
# 批量转码（目录递归、镜像子目录、默认 on-exist=skip）
python -m bvp batch ./media -o ./out -j 2

# 指定编码/容器与自定义扩展名
python -m bvp batch ./media -o ./out --codec h265 --container mkv --ext .mp4 --ext .mov

# 独立对账（不运行转码），退出码 0 通过、8 不符
python -m bvp verify ./out
```

退出码：`0` 全部成功/跳过且对账通过；`8` 存在失败条目或对账不符；
`2` 启动前参数错误。

## 作为库使用

```python
from bvp import batch_transcode

result = batch_transcode(["./media"], output_dir="./out", jobs=2)
print(result.summary)                 # {'total': n, 'success': .., 'failed': .., 'skipped': ..}
print(result.reconciliation.ok)       # 清单与磁盘是否对得上
```

## 测试

```bash
python -m pytest tests/ -q
```

- 纯逻辑测试（契约/扫描/规划/清单/复核/批量注入）不依赖 ffmpeg；
- `tests/test_integration.py` 为真实 ffmpeg 端到端用例，无 ffmpeg 时自动 skip；
- 批量层通过 `transcode_fn` 注入假核心，可在无 ffmpeg 环境覆盖全部容错路径。
