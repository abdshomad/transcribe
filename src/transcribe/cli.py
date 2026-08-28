import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from .pipeline import AudioTranscriptionPipeline
from .downloader import is_url, is_gdrive_folder, fetch_gdrive_folder_contents
from .models import TranscriptSegment, MODEL_CATALOG, TranscriptionResult, DiarizedSegment
from .youtube import is_youtube_url, fetch_youtube_transcript
from .exporters import export_transcription
from .history import save_history, find_checkpoint
from .merger import (
    sort_media_files_by_sequence,
    combine_transcription_results,
    export_all_combined_formats,
)

app = typer.Typer(
    name="transcribe",
    help="Fast speech-to-text transcription, model catalog, and multi-model benchmarks.",
    add_completion=False,
)
console = Console()


def _sanitize_dirname(name: str) -> str:
    """Sanitize folder or file name for filesystem safety."""
    import re
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def _render_live_segment(seg: TranscriptSegment) -> None:
    """Print streaming transcript segment with timestamp."""
    start_fmt = f"{int(seg.start // 60):02d}:{seg.start % 60:05.2f}"
    end_fmt = f"{int(seg.end // 60):02d}:{seg.end % 60:05.2f}"
    console.print(f"  [cyan]{start_fmt} ➜ {end_fmt}[/cyan]  {seg.text.strip()}")


def _render_files_table(exported: dict[str, str], title: str = "\nGenerated Output Files") -> None:
    """Display table of exported files."""
    table = Table(title=title, border_style="green")
    table.add_column("Format", style="cyan", justify="center")
    table.add_column("File Path", style="magenta")
    for fmt, path in exported.items():
        table.add_row(fmt.upper(), path)
    console.print(table)


def _handle_youtube_run(audio_source: str, output_dir: Path, formats: List[str], force: bool) -> None:
    """Handle YouTube URL transcription or caption extraction."""
    console.print(Panel.fit(
        f"[bold white]YouTube Transcription Mode[/bold white]\n"
        f"URL: [cyan]{audio_source}[/cyan]" + (" | [yellow]FORCE MODE[/yellow]" if force else ""),
        border_style="red",
    ))
    console.print("[bold cyan]Searching for existing YouTube subtitles/captions via yt-dlp...[/bold cyan]")
    yt_res = fetch_youtube_transcript(audio_source)
    if not yt_res:
        console.print("[bold yellow]Sorry, no existing transcription or subtitles found for this YouTube video.[/bold yellow]")
        return

    console.print(f"[bold green]✔ Found existing subtitles for:[/bold green] [yellow]{yt_res['title']}[/yellow] ({yt_res['duration']:.1f}s, lang: {yt_res['language']})\n")
    for s in yt_res["segments"]:
        _render_live_segment(s)

    diarized_segs = [
        DiarizedSegment(
            id=s.id,
            speaker="SPEAKER_00",
            start=s.start,
            end=s.end,
            text=s.text,
            words=s.words
        )
        for s in yt_res["segments"]
    ]
    trans_result = TranscriptionResult(
        language=yt_res["language"],
        language_probability=1.0,
        duration=yt_res["duration"],
        segments=diarized_segs,
        speakers=["SPEAKER_00"]
    )
    exported = export_transcription(trans_result, output_dir, f"youtube_{yt_res['video_id']}", formats)
    job_id = f"yt_{yt_res['video_id']}_{int(time.time())}" if force else f"yt_{yt_res['video_id']}"
    save_history(
        job_id=job_id,
        source_name=yt_res["title"],
        model="youtube-captions",
        result_data=trans_result.model_dump(),
        status="completed",
        last_processed_time=yt_res["duration"],
        processing_time=0.5,
        audio_path=audio_source
    )
    _render_files_table(exported)
    console.print("[bold green]✔ Done![/bold green]\n")


def _make_cli_progress_handler() -> Callable[[dict], None]:
    """Create a progress reporter for CLI media conversion and downloading."""
    last_reported = {"converting": -1, "downloading": -1}

    def on_prog(info: dict) -> None:
        stage = info.get("stage")
        if stage in last_reported:
            pct = int(info.get("percent", 0.0))
            step = 10 if stage == "converting" else 25
            if pct != last_reported[stage] and (pct % step == 0 or pct == 100):
                last_reported[stage] = pct
                icon = "⚙️ Converting media..." if stage == "converting" else "📥 Downloading..."
                console.print(f"  [cyan]{icon} {pct}%[/cyan]")

    return on_prog


def _run_single_file(
    pipeline: AudioTranscriptionPipeline,
    audio_source: str,
    display_name: str,
    output_dir: Path,
    formats: List[str],
    language: Optional[str],
    num_speakers: Optional[int],
    force: bool,
    output_stem: Optional[str] = None,
) -> dict[str, str]:
    """Execute single audio transcription with silent checkpoint recovery."""
    start_offset = 0.0
    existing_segments = None
    if not force:
        cp = find_checkpoint(display_name, pipeline.transcriber.model_name)
        if cp and cp.get("last_processed_time", 0) > 0:
            start_offset = float(cp["last_processed_time"])
            raw_segs = cp.get("segments", [])
            existing_segments = [TranscriptSegment(**s) for s in raw_segs]
            start_fmt = f"{int(start_offset // 60):02d}:{start_offset % 60:05.2f}"
            console.print(f"  [bold yellow]⚡ Checkpoint detected! Auto-resuming from {start_fmt}...[/bold yellow]")

    start_t = time.time()
    prog_handler = _make_cli_progress_handler()
    res = pipeline.process(
        audio_path_or_url=audio_source,
        language=language,
        num_speakers=num_speakers,
        start_offset=start_offset,
        existing_segments=existing_segments,
        on_segment=_render_live_segment,
        on_progress=prog_handler,
    )
    stem = output_stem or Path(display_name).stem or "transcript"
    exported = export_transcription(res, output_dir, stem=stem, formats=formats)
    elapsed = time.time() - start_t

    job_id = f"cli_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    save_history(
        job_id=job_id,
        source_name=display_name,
        model=pipeline.transcriber.model_name,
        result_data=res.model_dump(),
        status="completed",
        last_processed_time=res.duration,
        processing_time=round(elapsed, 2),
        audio_path=audio_source,
    )
    return exported, res


def _handle_batch_combination(
    results_with_metadata: List[tuple[str, TranscriptionResult]],
    folder_title: str,
    safe_folder: str,
    model: str,
    model_suffix: bool,
    batch_output_dir: Path,
    formats: List[str],
    pipeline: AudioTranscriptionPipeline,
    audio_source: str,
) -> None:
    """Stitch sequential parts into unified continuous transcript and register in history."""
    console.print("\n[bold cyan]🧩 Stitching continuous folder transcription across sequence...[/bold cyan]")
    try:
        combined_res = combine_transcription_results(results_with_metadata, folder_title)
        comb_stem = f"{safe_folder}_combined_{model}" if model_suffix else f"{safe_folder}_combined"
        comb_exported = export_all_combined_formats(combined_res, batch_output_dir, comb_stem, formats, folder_title)

        job_id = f"cli_combined_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        total_proc_time = sum(getattr(r, "processing_time", 0.0) or 0.0 for _, r in results_with_metadata)
        save_history(
            job_id=job_id,
            source_name=folder_title,
            model=pipeline.transcriber.model_name,
            result_data=combined_res.model_dump(),
            status="completed",
            last_processed_time=combined_res.duration,
            processing_time=round(total_proc_time, 2) if total_proc_time else round(combined_res.duration, 2),
            audio_path=audio_source,
        )
        _render_files_table({k: str(v) for k, v in comb_exported.items()}, title="🧩 Unified Continuous Transcript Files")
    except Exception as e:
        console.print(f"[bold yellow]⚠️ Notice: Could not combine folder results: {e}[/bold yellow]")


def _process_batch_items(
    media_files: List[dict],
    model: str,
    model_suffix: bool,
    pipeline: AudioTranscriptionPipeline,
    batch_output_dir: Path,
    formats: List[str],
    language: Optional[str],
    num_speakers: Optional[int],
    force: bool,
) -> tuple[List[tuple], List[tuple[str, TranscriptionResult]]]:
    """Iterate through media files and transcribe each part."""
    summary_rows: List[tuple] = []
    results_with_metadata: List[tuple[str, TranscriptionResult]] = []
    for idx, f in enumerate(media_files, start=1):
        console.print(Panel.fit(
            f"[bold white]Batch Item [{idx}/{len(media_files)}][/bold white]: [cyan]{f['name']}[/cyan] ([yellow]{model}[/yellow])",
            border_style="blue",
        ))
        t0 = time.time()
        base_stem = Path(f["name"]).stem or f["name"]
        stem = f"{base_stem}_{model}" if model_suffix else base_stem
        try:
            _, res = _run_single_file(
                pipeline=pipeline,
                audio_source=f["url"],
                display_name=f["name"],
                output_dir=batch_output_dir,
                formats=formats,
                language=language,
                num_speakers=num_speakers,
                force=force,
                output_stem=stem,
            )
            results_with_metadata.append((f["name"], res))
            elapsed = time.time() - t0
            summary_rows.append((idx, f["name"], "[green]Success[/green]", f"{elapsed:.1f}s"))
        except Exception as e:
            elapsed = time.time() - t0
            console.print(f"[bold red]❌ Failed to transcribe '{f['name']}': {e}[/bold red]")
            summary_rows.append((idx, f["name"], f"[red]Failed ({type(e).__name__})[/red]", f"{elapsed:.1f}s"))
    return summary_rows, results_with_metadata


def _run_gdrive_folder_batch(
    audio_source: str,
    output_dir: Path,
    model: str,
    device: str,
    language: Optional[str],
    num_speakers: Optional[int],
    diarize: bool,
    formats: List[str],
    force: bool,
    yes: bool,
    model_suffix: bool = False,
    combine: bool = True,
) -> None:
    """Discover files in Google Drive folder, sort by sequence, process and optionally stitch."""
    console.print("[bold cyan]🔍 Discovering audio recordings inside Google Drive folder...[/bold cyan]")
    folder_title, raw_files = fetch_gdrive_folder_contents(audio_source)
    if not raw_files:
        console.print("[bold yellow]No audio/video recordings found in this Google Drive folder.[/bold yellow]")
        return

    media_files = sort_media_files_by_sequence(raw_files)
    table = Table(title=f"📁 Discovered & Sequenced Files in: {folder_title}", border_style="cyan")
    table.add_column("#", justify="center", style="bold cyan")
    table.add_column("Recording Name", style="bold yellow")
    table.add_column("File ID", style="magenta")
    for idx, f in enumerate(media_files, start=1):
        table.add_row(str(idx), f["name"], f["id"])
    console.print(table)

    if not yes:
        confirmed = typer.confirm(f"\nProceed with batch transcribing all {len(media_files)} files?", default=True)
        if not confirmed:
            console.print("[yellow]Batch processing cancelled by user.[/yellow]")
            return

    safe_folder = _sanitize_dirname(folder_title)
    batch_output_dir = output_dir / safe_folder
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = AudioTranscriptionPipeline(
        whisper_model_size=model,
        device=device,
        enable_diarization=diarize,
    )

    summary_rows, results_with_metadata = _process_batch_items(
        media_files=media_files,
        model=model,
        model_suffix=model_suffix,
        pipeline=pipeline,
        batch_output_dir=batch_output_dir,
        formats=formats,
        language=language,
        num_speakers=num_speakers,
        force=force,
    )

    # Render final batch summary
    summary_table = Table(title="\n📊 Google Drive Batch Summary", border_style="green")
    summary_table.add_column("#", justify="center", style="cyan")
    summary_table.add_column("Recording Name", style="bold yellow")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Time", justify="right", style="magenta")
    for r in summary_rows:
        summary_table.add_row(str(r[0]), r[1], r[2], r[3])
    console.print(summary_table)

    if combine and len(results_with_metadata) > 1:
        _handle_batch_combination(
            results_with_metadata=results_with_metadata,
            folder_title=folder_title,
            safe_folder=safe_folder,
            model=model,
            model_suffix=model_suffix,
            batch_output_dir=batch_output_dir,
            formats=formats,
            pipeline=pipeline,
            audio_source=audio_source,
        )

    console.print(f"[bold green]✔ Batch complete! Results saved in: [cyan]{batch_output_dir}[/cyan][/bold green]\n")


@app.command("run")
def transcribe_cmd(
    audio_source: str = typer.Argument(..., help="Local file path, YouTube URL, or Google Drive / HTTP URL"),
    output_dir: Path = typer.Option(Path("./output"), "--output-dir", "-o", help="Output directory"),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model (e.g. tiny, base, small, medium, large-v3, turbo)"),
    device: str = typer.Option("auto", "--device", "-d", help="Device to use (auto, cuda, cpu)"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (e.g. en, id, es)"),
    num_speakers: Optional[int] = typer.Option(None, "--num-speakers", "-s", help="Exact number of speakers"),
    diarize: bool = typer.Option(False, "--diarize/--no-diarize", help="Enable speaker diarization"),
    formats: List[str] = typer.Option(["json", "txt", "srt", "vtt"], "--format", "-f", help="Output formats"),
    force: bool = typer.Option(False, "--force", help="Force re-transcribe and record a fresh run in history"),
    yes: bool = typer.Option(True, "--yes/--prompt", "-y", help="Proceed automatically with batch processing (default: auto-proceed)"),
    model_suffix: bool = typer.Option(False, "--model-suffix", help="Append model name as suffix to output filenames"),
    combine: bool = typer.Option(True, "--combine/--no-combine", help="Stitch sequential folder recordings into one continuous transcript"),
):
    """Transcribe an audio file, YouTube URL, or Google Drive folder with stream output."""
    if is_youtube_url(audio_source):
        _handle_youtube_run(audio_source, output_dir, formats, force)
        return

    if is_gdrive_folder(audio_source):
        _run_gdrive_folder_batch(
            audio_source=audio_source,
            output_dir=output_dir,
            model=model,
            device=device,
            language=language,
            num_speakers=num_speakers,
            diarize=diarize,
            formats=formats,
            force=force,
            yes=yes,
            model_suffix=model_suffix,
            combine=combine,
        )
        return

    if not is_url(audio_source) and not Path(audio_source).exists():
        console.print(f"[bold red]Error:[/bold red] Local file not found: {audio_source}")
        raise typer.Exit(code=1)

    display_name = audio_source if is_url(audio_source) else Path(audio_source).name
    force_tag = " | [bold yellow]FORCE RE-TRANSCRIBE[/bold yellow]" if force else ""
    console.print(Panel.fit(
        f"[bold white]Audio Transcription Engine[/bold white]\n"
        f"Source: [cyan]{display_name}[/cyan] | Model: [yellow]{model}[/yellow] | Device: [yellow]{device}[/yellow]{force_tag}",
        border_style="blue",
    ))

    pipeline = AudioTranscriptionPipeline(
        whisper_model_size=model,
        device=device,
        enable_diarization=diarize,
    )

    console.print("[bold green]Transcribing speech stream:[/bold green]")
    exported, _ = _run_single_file(
        pipeline=pipeline,
        audio_source=audio_source,
        display_name=display_name,
        output_dir=output_dir,
        formats=formats,
        language=language,
        num_speakers=num_speakers,
        force=force,
    )

    _render_files_table(exported)
    console.print("[bold green]✔ Done![/bold green]\n")


@app.command("models")
def list_models_cmd(
    family: Optional[str] = typer.Option(None, "--family", "-f", help="Filter by family (Standard, English, Distil, Turbo)"),
):
    """List all available ASR / STT models with parameters, speed factors, and requirements."""
    table = Table(title="🎙️ Available ASR / STT Models", border_style="cyan")
    table.add_column("Model Name", style="bold yellow", justify="left")
    table.add_column("Family", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("Speed", style="green", justify="right")
    table.add_column("Languages", style="magenta")
    table.add_column("Description", style="slate38" if "slate" in str(console) else "white")

    for m in MODEL_CATALOG:
        if family and family.lower() not in m.family.lower():
            continue
        table.add_row(m.name, m.family, m.params, m.vram, m.speed_factor, m.languages, m.description)

    console.print(table)


@app.command("benchmark")
def benchmark_models_cmd(
    audio_source: str = typer.Argument(..., help="Audio file path to benchmark across models"),
    models: List[str] = typer.Option(["tiny", "base", "small"], "--models", "-m", help="Models to benchmark sequentially"),
    device: str = typer.Option("auto", "--device", "-d", help="Device to use (auto, cuda, cpu)"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code"),
):
    """Sequentially benchmark multiple ASR models on an audio file and print comparative performance."""
    if not Path(audio_source).exists():
        console.print(f"[bold red]Error:[/bold red] Audio file not found: {audio_source}")
        raise typer.Exit(code=1)

    console.print(Panel.fit(
        f"[bold white]ASR Sequential Benchmark[/bold white]\n"
        f"File: [cyan]{Path(audio_source).name}[/cyan] | Models: [yellow]{', '.join(models)}[/yellow]",
        border_style="magenta",
    ))

    results_table = Table(title="\n📊 Sequential Benchmark Results", border_style="green")
    results_table.add_column("Model", style="bold yellow")
    results_table.add_column("Duration", justify="right")
    results_table.add_column("Processing Time", justify="right", style="cyan")
    results_table.add_column("Speed Factor (RTF)", justify="right", style="green")
    results_table.add_column("Segments", justify="right")
    results_table.add_column("Words", justify="right")
    results_table.add_column("Status", justify="center")

    for model_name in models:
        console.print(f"\n[bold blue]▶ Testing Model [{model_name}]...[/bold blue]")
        t0 = time.time()
        try:
            pipeline = AudioTranscriptionPipeline(whisper_model_size=model_name, device=device, enable_diarization=False)
            res = pipeline.process(audio_path_or_url=audio_source, language=language)
            elapsed = time.time() - t0
            dur = res.duration or 1.0
            rtf = f"{dur / elapsed:.1f}x"
            word_count = sum(len(s.text.split()) for s in res.segments)
            results_table.add_row(
                model_name,
                f"{dur:.1f}s",
                f"{elapsed:.2f}s",
                rtf,
                str(len(res.segments)),
                str(word_count),
                "[bold green]✔ Passed[/bold green]",
            )
        except Exception as e:
            elapsed = time.time() - t0
            results_table.add_row(
                model_name,
                "--",
                f"{elapsed:.2f}s",
                "--",
                "--",
                "--",
                f"[bold red]✘ {str(e)[:20]}[/bold red]",
            )

    console.print(results_table)
    console.print("[bold green]✔ Sequential benchmark completed![/bold green]\n")


@app.command("test-all")
def test_all_cmd(
    audio_source: str = typer.Option("data/sample/proklamasi.wav", "--audio", "-a", help="Audio file path to benchmark"),
    output_dir: str = typer.Option("tests/each-model", "--output-dir", "-o", help="Base output directory"),
    models: Optional[List[str]] = typer.Option(None, "--models", "-m", help="Specific models to test sequentially (defaults to local catalog)"),
    retry_failed: bool = typer.Option(False, "--retry-failed", "-r", help="Only retry models that previously failed or are missing"),
    language: Optional[str] = typer.Option("id", "--language", "-l", help="Language code"),
    device: str = typer.Option("auto", "--device", "-d", help="Device (auto, cuda, cpu)"),
):
    """Sequentially test all ASR models and save individual results into tests/each-model/{model}/."""
    import json
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if retry_failed:
        failed = []
        for item in sorted(base_dir.iterdir()):
            if item.is_dir():
                mf = item / "metrics.json"
                if not mf.exists():
                    failed.append(item.name)
                else:
                    try:
                        data = json.loads(mf.read_text(encoding="utf-8"))
                        if data.get("status") == "failed":
                            failed.append(data.get("model_name", item.name))
                    except Exception:
                        failed.append(item.name)
        if not failed:
            console.print("[bold green]✔ No failed models found in output directory! All models passed.[/bold green]")
            return
        target_models = failed
        console.print(f"[bold yellow]🔄 Retrying {len(target_models)} failed models:[/bold yellow] {', '.join(target_models)}")
    else:
        target_models = models or ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "distil-small.en", "medium", "medium.en", "distil-medium.en", "large-v3", "turbo", "cahya-whisper-medium-id"]

    console.print(Panel.fit(
        f"[bold white]Sequential Model Evaluation Suite[/bold white]\n"
        f"Audio: [cyan]{audio_source}[/cyan] | Output Dir: [magenta]{output_dir}[/magenta] | Target Models: [yellow]{len(target_models)}[/yellow]",
        border_style="magenta",
    ))

    summary_table = Table(title="\n📊 Sequential Model Evaluation Summary", border_style="green")
    summary_table.add_column("Model Name", style="bold yellow")
    summary_table.add_column("Duration", justify="right")
    summary_table.add_column("Processing Time", justify="right", style="cyan")
    summary_table.add_column("Speed Factor", justify="right", style="green")
    summary_table.add_column("Words", justify="right")
    summary_table.add_column("Status", justify="center")

    for idx, model_name in enumerate(target_models, 1):
        clean_folder = model_name.replace("/", "-").replace(".", "_")
        model_dir = base_dir / clean_folder
        model_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold blue][{idx}/{len(target_models)}] Testing model [{model_name}]...[/bold blue]")
        t0 = time.time()
        metrics = {
            "model_name": model_name,
            "audio_file": str(audio_source),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            pipeline = AudioTranscriptionPipeline(whisper_model_size=model_name, device=device, enable_diarization=False)
            res = pipeline.process(audio_path_or_url=audio_source, language=language)
            elapsed = time.time() - t0
            dur = res.duration or 1.0
            rtf = dur / elapsed if elapsed > 0 else 0.0
            word_count = sum(len(s.text.split()) for s in res.segments)

            transcript_text = "\n".join(f"[{s.start:.2f}s - {s.end:.2f}s] {s.text.strip()}" for s in res.segments)
            (model_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")

            metrics.update({
                "status": "passed",
                "audio_duration_seconds": dur,
                "processing_time_seconds": round(elapsed, 3),
                "speed_factor_rtf": f"{rtf:.2f}x",
                "total_segments": len(res.segments),
                "total_words": word_count,
                "detected_language": res.language,
                "language_probability": res.language_probability,
            })
            (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            summary_table.add_row(
                model_name,
                f"{dur:.1f}s",
                f"{elapsed:.2f}s",
                f"{rtf:.1f}x",
                str(word_count),
                "[bold green]✔ Passed[/bold green]",
            )
            console.print(f"  [green]✔ Saved to {model_dir}/ (elapsed: {elapsed:.2f}s, speed: {rtf:.1f}x)[/green]")
        except Exception as e:
            elapsed = time.time() - t0
            err_msg = str(e)
            metrics.update({
                "status": "failed",
                "processing_time_seconds": round(elapsed, 3),
                "error": err_msg,
            })
            (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            (model_dir / "transcript.txt").write_text(f"ERROR: {err_msg}\n", encoding="utf-8")

            summary_table.add_row(
                model_name,
                "--",
                f"{elapsed:.2f}s",
                "--",
                "--",
                f"[bold red]✘ Failed[/bold red]",
            )
            console.print(f"  [red]✘ Failed on {model_name}: {err_msg[:60]}[/red]")

    console.print(summary_table)
    console.print(f"\n[bold green]✔ Done! All results saved to {output_dir}/[/bold green]\n")


def main():
    """CLI Entry point."""
    app()


if __name__ == "__main__":
    main()


