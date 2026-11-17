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
| `bvp/rules/linemap.py` | 配置文本的 JSON 路径→行号定位、重复键检测 |
| `bvp/rules/schema.py` | v1 规则模式、静态值域与**全局可满足性校验** |
| `bvp/rules/engine.py` | 规则求值：匹配/继承/覆盖/组合收敛/解释轨迹 |
| `bvp/rules/store.py` | 不可变快照 + 原子热加载 `ConfigStore` |
| `bvp/cli.py` | 命令行入口 `python -m bvp` |

## 规则引擎：把文件解析为确定、可解释的参数集

`--rules rules.json` 启用规则引擎。规则不只是「匹配哪些文件」，而是把每个
入选文件收敛为**唯一一组完整转码参数**（编码/容器/CRF/码率/分辨率/策略/
输出目录），并能解释每个参数来自哪条规则。

### 配置文件（JSON v1，纯静态规则语言）

```json
{
  "version": 1,
  "defaults": { "crf": 26, "preset": "fast", "jobs": 2 },
  "rules": [
    { "name": "hevc-base", "abstract": true,
      "params": { "codec": "h265", "container": "mkv" } },
    { "name": "archive", "dir": "./archive", "ext": [".mp4", ".mov"],
      "inherits": ["hevc-base"], "params": { "crf": 24 } },
    { "name": "web", "priority": 10, "dir": "./archive/web",
      "params": { "codec": "h264", "resolution": "720p" } },
    { "name": "skip-temp", "dir": "./archive",
      "exclude": ["*.tmp.mp4", "drafts/*"] }
  ]
}
```

- **匹配条件**（闭合集合）：`dir`（相对路径以配置文件目录为基准）、
  `ext`（小写含点，多值 OR）、`exclude`（glob，含分隔符按子树匹配）；
  三个维度之间取 AND。
- **继承**：`inherits` 只继承基类的 `params`（不继承匹配条件）；抽象规则
  （`abstract: true`）不直接匹配，仅用于被继承。
- **参数键**（闭合集合）：`codec` / `container` / `crf` / `preset` /
  `audio_bitrate` / `resolution` / `on_exist` / `output_dir`；`defaults`
  额外允许 `jobs` / `recursive`。出现未知键即拒绝。

### 求值顺序与冲突解决

对任意文件，过程完全确定（与枚举次序无关）：

1. **排除优先**：任一规则的 `exclude` 命中即硬过滤，不产生清单条目；
2. **收集命中**：具体规则按 `(priority 降序, 定义顺序)` 求值；
3. **继承展开**：基类参数先合并，自身参数覆盖；
4. **覆盖收敛**：命中规则从低优先级到高优先级逐层合并，高优先级逐字段
   覆盖，最终每个字段都有确定值；
5. **缺省兜底**：无规则命中时落回隐式 `__default__`（defaults + 内置默认）。

合并优先级（高 → 低）：**CLI 显式选项 → 命中规则参数 → 配置 defaults →
内置默认值**。CLI 用 `argparse.SUPPRESS` 区分「显式给出」与「未指定」，
未指定的选项不会覆盖规则。

### 全局可满足性校验（启动期，非逐字段）

校验在任何扫描/转码之前执行，**一次性收齐全部问题并定位到具体行**，任一
问题即拒绝启动（退出码 2）：

- **矛盾参数**：两条同优先级、可能命中同一文件的规则对同一参数给出不同
  字面值（不同优先级不是矛盾——高优先级覆盖语义明确）；
- **不支持的编码器**：codec 不在构建支持集合，或本机 ffmpeg 不提供该编码器
  （本机探测可注入，失败时降级为构建级校验）；
- **属性依赖**：参数取值依赖被匹配文件的运行期属性（分辨率/时长），如
  `"${width/2}"`、`"=..."` 等表达式一律拒绝——v1 只接受静态字面量，保证
  「任意输入会产出什么参数」在启动前可完整推理；
- 另有 JSON 语法、重复键、悬空/循环继承、值域越界（crf 0-51 等）、
  分辨率表达式语法等结构与值域检查。

### 原子热加载

`ConfigStore` 持有不可变 `RuleSet` 快照：

- 切换是锁内一次引用替换，读取永远拿到完整一致的配置，**无中间态**；
- 批量任务启动时取走快照引用，整个生命周期持有该不可变对象——配置替换
  不影响进行中任务（旧配置），只有新任务用新快照；
- 重新加载校验失败时**保留现行快照**继续服务，错误上报而非污染；
- 轮询基于 `(mtime_ns, size)`，配置文件应以「同目录临时文件 + `os.replace`」
  维护原子性。CLI 用 `--watch --poll-interval N --max-cycles K` 驱动。

### 规则相关命令

```bash
# 只校验配置并打印规则清单与示例文件的求值解释，不转码
python -m bvp rules-check rules.json ./archive/a.mp4

# 规则驱动批量
python -m bvp batch ./archive --rules rules.json -o ./out

# CLI 显式选项压过规则（仅覆盖给出的字段，其余仍由规则决定）
python -m bvp batch ./archive --rules rules.json --codec h264

# 热加载模式：每轮批量后检查配置变化并原子切换
python -m bvp batch ./archive --rules rules.json --watch --poll-interval 2
```

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

# 规则引擎：逐文件按配置裁决参数（继承/覆盖/排除/全局校验/热加载，见下文）
python -m bvp batch ./media --rules rules.json -o ./out
python -m bvp rules-check rules.json            # 只校验不转码，带行号与求值解释

# 独立对账（不运行转码），退出码 0 通过、8 不符
python -m bvp verify ./out
```

退出码：`0` 全部成功/跳过且对账通过；`8` 存在失败条目或对账不符；
`2` 启动前参数错误（含规则配置校验失败，错误定位到具体行）。

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
