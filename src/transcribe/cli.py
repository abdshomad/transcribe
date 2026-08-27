import time
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from .pipeline import AudioTranscriptionPipeline
from .downloader import is_url
from .models import TranscriptSegment, MODEL_CATALOG
from .youtube import is_youtube_url, fetch_youtube_transcript
from .exporters import export_transcription
from .history import save_history

app = typer.Typer(
    name="transcribe",
    help="Fast speech-to-text transcription, model catalog, and multi-model benchmarks.",
    add_completion=False,
)
console = Console()


def _render_live_segment(seg: TranscriptSegment) -> None:
    """Print streaming transcript segment with timestamp."""
    start_fmt = f"{int(seg.start // 60):02d}:{seg.start % 60:05.2f}"
    end_fmt = f"{int(seg.end // 60):02d}:{seg.end % 60:05.2f}"
    console.print(f"  [cyan]{start_fmt} ➜ {end_fmt}[/cyan]  {seg.text.strip()}")


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
):
    """Transcribe an audio file or YouTube URL with real-time stream output."""
    import uuid
    # Check if input is a YouTube URL
    if is_youtube_url(audio_source):
        console.print(Panel.fit(
            f"[bold white]YouTube Transcription Mode[/bold white]\n"
            f"URL: [cyan]{audio_source}[/cyan]" + (" | [yellow]FORCE MODE[/yellow]" if force else ""),
            border_style="red",
        ))
        console.print("[bold cyan]Searching for existing YouTube subtitles/captions via yt-dlp...[/bold cyan]")
        yt_res = fetch_youtube_transcript(audio_source)
        if yt_res:
            console.print(f"[bold green]✔ Found existing subtitles for:[/bold green] [yellow]{yt_res['title']}[/yellow] ({yt_res['duration']:.1f}s, lang: {yt_res['language']})\n")
            for s in yt_res["segments"]:
                _render_live_segment(s)
                
            from .models import TranscriptionResult, DiarizedSegment
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
            
            table = Table(title="\nGenerated Output Files", border_style="green")
            table.add_column("Format", style="cyan", justify="center")
            table.add_column("File Path", style="magenta")
            for fmt, path in exported.items():
                table.add_row(fmt.upper(), path)
            console.print(table)
            console.print("[bold green]✔ Done![/bold green]\n")
            return
        else:
            console.print("[bold yellow]Sorry, no existing transcription or subtitles found for this YouTube video.[/bold yellow]")
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
    exported = pipeline.process_and_export(
        audio_path_or_url=audio_source,
        output_dir=output_dir,
        formats=formats,
        language=language,
        num_speakers=num_speakers,
        on_segment=_render_live_segment,
    )

    table = Table(title="\nGenerated Output Files", border_style="green")
    table.add_column("Format", style="cyan", justify="center")
    table.add_column("File Path", style="magenta")
    for fmt, path in exported.items():
        table.add_row(fmt.upper(), path)

    console.print(table)
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


