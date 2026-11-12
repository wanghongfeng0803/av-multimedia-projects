"""VFR -> CFR transcoder service.

Components:
    ffmpeg_utils : low level ffmpeg/ffprobe subprocess wrappers
    media        : source analysis (time base, dominant fps, timestamps)
    pipeline     : segmented, resumable VFR->CFR transcode pipeline
    state        : SQLite-backed job state machine and worker lease
    worker       : poller that drives jobs through the pipeline
    api          : FastAPI HTTP surface
    cli          : standalone command line entry point
"""

__version__ = "1.0.0"
