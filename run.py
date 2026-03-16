from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch


def log_mem(label, logger=None):
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    msg = f"[MEM {label}] alloc={alloc:.2f}GB  peak={peak:.2f}GB  reserved={reserved:.2f}GB"
    if logger:
        logger.info(msg)
    else:
        print(msg)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.utils.profiler import get_profiler, profiler_section
    from vipe.streams.base import StreamList

    profiler_cfg = getattr(args, "profiler", None)
    profiler = get_profiler()
    profiler.reset()

    profiling_enabled = bool(getattr(profiler_cfg, "enabled", False)) if profiler_cfg is not None else False
    if profiling_enabled:
        profiler.enable()
    else:
        profiler.disable()

    memory_profiling = getattr(args, "memory_profiler", False)
    memory_snapshot_dir = Path(getattr(args, "memory_snapshot_dir", "memory_snapshots"))

    # Gather all video streams
    stream_list = StreamList.make(args.streams)
    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # Process each video stream
    logger = configure_logging()

    if memory_profiling:
        memory_snapshot_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CUDA memory profiling enabled, snapshots will be saved to %s", memory_snapshot_dir)

    with profiler_section("Vipe"):
        for stream_idx in range(len(stream_list)):
            video_stream = stream_list[stream_idx]
            logger.info(
                f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})"
            )
            pipeline = make_pipeline(args.pipeline)

            if memory_profiling:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.memory._record_memory_history()
                log_mem(f"before:{video_stream.name()}", logger)

            try:
                with profiler_section(f"pipeline.run[{video_stream.name()}]"):
                    pipeline.run(video_stream)
            except torch.cuda.OutOfMemoryError:
                snapshot_path = memory_snapshot_dir / f"oom_{video_stream.name()}.pkl"
                snapshot = torch.cuda.memory._snapshot()
                lightweight = {
                    "segments": snapshot.get("segments", []),
                    "device_traces": snapshot.get("device_traces", []),
                }
                snapshot = lightweight
                from pickle import dump
                with open(snapshot_path, "wb") as f:
                    dump(snapshot, f)
                logger.error(
                    "OOM during %s — snapshot saved to %s",
                    video_stream.name(), snapshot_path,
                )
                logger.error("Memory state at OOM:\n%s", torch.cuda.memory_summary())
                raise
            finally:
                if memory_profiling:
                    log_mem(f"after:{video_stream.name()}", logger)
                    # Save a snapshot for every stream, not just OOM
                    snapshot_path = memory_snapshot_dir / f"{video_stream.name()}.pkl"
                    snapshot = torch.cuda.memory._snapshot()
                    lightweight = {
                        "segments": snapshot.get("segments", []),
                        "device_traces": snapshot.get("device_traces", []),
                    }
                    snapshot = lightweight
                    from pickle import dump
                    with open(snapshot_path, "wb") as f:
                        dump(snapshot, f)
                    logger.info("Memory snapshot saved to %s", snapshot_path)
                    torch.cuda.memory._record_memory_history(enabled=None)

            logger.info(f"Finished processing {video_stream.name()}")

    if memory_profiling:
        logger.info("Final memory state:\n%s", torch.cuda.memory_summary())
        logger.info(
            "Visualize snapshots with: python -m torch.utils.viz._memory_viz trace_plot <snapshot.pkl> -o trace.html"
        )

    if profiling_enabled:
        min_percentage = float(getattr(profiler_cfg, "min_percentage", 0.0)) if profiler_cfg is not None else 0.0
        max_depth_cfg = getattr(profiler_cfg, "max_depth", None) if profiler_cfg is not None else None
        max_depth = int(max_depth_cfg) if max_depth_cfg not in (None, "null") else None
        report = profiler.report(min_percentage=min_percentage, max_depth=max_depth)
        output_path = getattr(profiler_cfg, "output", None) if profiler_cfg is not None else None
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report)
            logger.info("Profiler report written to %s", path.resolve())
        logger.info("Profiling summary:\n%s", report)


if __name__ == "__main__":
    run()