# VFR 转码器（作业 05）

处理 VFR（时间基 `1/1000`、带 B 帧重排）老素材的转码服务：输出 CFR、音画偏移
≤ 40 ms（3 小时素材首尾都测）、保留 copyts 语义，提供 HTTP API + worker 进程池、
任务状态机与 kill 后断点续跑。

完整实现、设计说明与实测数据见 **[`vfr-transcoder/README.md`](vfr-transcoder/README.md)**。

快速入口：

```bash
cd vfr-transcoder
pip install -r requirements.txt

# 命令行
python -m vfr_transcoder transcode input.mkv out.mp4 --target-fps 25
# HTTP 服务 + worker
python -m vfr_transcoder serve --port 8080
python -m vfr_transcoder worker
# 测试（含 kill -9 续跑、API+worker 集成）
python -m unittest discover -s tests -v
# 长片门禁（30 分钟起，3 小时用 --duration 10800）
python tools/run_long_gate.py --work-dir ./work/long --duration 1800
```
