"""VideoAnnotator CLI - Unified command-line interface."""

import os
from pathlib import Path

import typer
import uvicorn

from .config_env import MAX_CONCURRENT_JOBS, WORKER_POLL_INTERVAL
from .validation.emotion_validator import validate_emotion_file
from .version import __version__

app = typer.Typer(
    name="videoannotator",
    help=f"VideoAnnotator v{__version__} - Production-ready video annotation toolkit\n\n"
    "Default: Starts API server when run without a command (uv run videoannotator)",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Enable development mode: allows all CORS origins (*), disables auth",
    ),
):
    """Start the API server when no subcommand is provided.

    This makes `uv run videoannotator` behave like `uv run videoannotator server`
    with the recommended host and port.
    """
    # If a subcommand was invoked, do nothing here and let Typer handle it.
    if ctx.invoked_subcommand is not None:
        return

    # Launch server with recommended defaults
    # NOTE: calling server() directly will run uvicorn and block the process.
    server(host="0.0.0.0", port=18011, reload=False, workers=1, dev=dev)


@app.command()
def server(
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to"),
    port: int = typer.Option(18011, help="Port to bind the server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Enable development mode: allows all CORS origins (*), disables auth",
    ),
):
    """Start the VideoAnnotator API server.

    By default, the server runs with secure settings (authentication required,
    restricted CORS origins). For local development with web clients, common
    development ports (3000, 5173, 8080, etc.) are already allowed.

    If you need to allow ALL origins (e.g., for testing with a remote client),
    use --dev mode or set CORS_ORIGINS="*" environment variable.
    """
    # Check if port is already in use before starting
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        sock.close()
    except OSError:
        typer.echo("")
        typer.echo(f"[ERROR] Port {port} is already in use", err=True)
        typer.echo("")
        typer.echo("To find the VideoAnnotator server process:", err=True)
        typer.echo("  ps aux | grep videoannotator | grep -v grep", err=True)
        typer.echo("")
        typer.echo("To stop ALL VideoAnnotator processes:", err=True)
        typer.echo("  pkill -f videoannotator", err=True)
        typer.echo("")
        typer.echo("To stop a specific process:", err=True)
        typer.echo("  kill <PID>  # Use PID from ps command above", err=True)
        typer.echo("")
        typer.echo("Alternatively, start on a different port (18012-18020 are CORS-enabled):", err=True)
        typer.echo("  uv run videoannotator server --port 18012", err=True)
        typer.echo("")
        raise typer.Exit(code=1)

    typer.echo(f"[START] Starting VideoAnnotator API server on http://{host}:{port}")
    typer.echo(f"[INFO] API documentation available at http://{host}:{port}/docs")

    # Configure development mode if requested
    if dev:
    
        os.environ["CORS_ORIGINS"] = "*"
        os.environ["VIDEOANNOTATOR_REQUIRE_AUTH"] = "false"
        typer.echo("")
        typer.echo("[DEV MODE] Development mode enabled:")
        typer.echo("  - CORS: All origins allowed (*)")
        typer.echo("  - Authentication: Disabled")
        typer.echo("  - [WARNING] Do not use in production!")
        typer.echo("")

    try:
        uvicorn.run(
            "videoannotator.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers
            if not reload
            else 1,  # Reload doesn't work with multiple workers
            log_level="info",
        )
    except OSError as e:
        if e.errno == 98:  # Address already in use
            typer.echo("")
            typer.echo(f"[ERROR] Port {port} is already in use", err=True)
            typer.echo("")
            typer.echo("To find the process using this port:", err=True)
            typer.echo(f"  lsof -i :{port}", err=True)
            typer.echo("  OR", err=True)
            typer.echo(f"  ss -tlnp | grep :{port}", err=True)
            typer.echo("")
            typer.echo("To stop the process:", err=True)
            typer.echo(f"  pkill -f 'uvicorn.*videoannotator'  # Stop any VideoAnnotator server", err=True)
            typer.echo("  OR", err=True)
            typer.echo(f"  kill <PID>  # Replace <PID> with the process ID from lsof/ss", err=True)
            typer.echo("")
            typer.echo("Alternatively, start the server on a different port:", err=True)
            typer.echo(f"  uv run videoannotator server --port <PORT>", err=True)
            typer.echo("")
        else:
            typer.echo(f"[ERROR] Failed to start server: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"[ERROR] Failed to start server: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def process(
    video: Path = typer.Argument(..., help="Path to video file to process"),
    output: Path | None = typer.Option(None, help="Output directory for results"),
    pipelines: str | None = typer.Option(
        None, help="Comma-separated list of pipelines to run"
    ),
    config: Path | None = typer.Option(None, help="Path to configuration file"),
):
    """Process a single video file (legacy mode)."""
    typer.echo(f"[PROCESS] Processing video: {video}")

    if not video.exists():
        typer.echo(f"[ERROR] Video file not found: {video}", err=True)
        raise typer.Exit(code=1)

    # TODO: Implement direct video processing using existing pipelines
    typer.echo("[WARNING] Direct processing not yet implemented in v1.2.0")
    typer.echo("[INFO] Use 'videoannotator server' and submit jobs via API")
    typer.echo("[INFO] See API docs at http://localhost:18011/docs")


@app.command()
def worker(
    poll_interval: int = typer.Option(
        WORKER_POLL_INTERVAL,
        help="Seconds between database polls (default: env WORKER_POLL_INTERVAL or 5)",
    ),
    max_concurrent: int = typer.Option(
        MAX_CONCURRENT_JOBS,
        help="Maximum concurrent jobs (default: env MAX_CONCURRENT_JOBS or 2)",
    ),
):
    """Start the background job processing worker.

    Environment variables:
        MAX_CONCURRENT_JOBS: Maximum number of jobs to process simultaneously (default: 2)
        WORKER_POLL_INTERVAL: Seconds between database polls (default: 5)
        MAX_JOB_RETRIES: Maximum retry attempts for failed jobs (default: 3)
        RETRY_DELAY_BASE: Base delay for exponential backoff (default: 2.0)
    """
    import asyncio

    from videoannotator.utils.logging_config import setup_videoannotator_logging
    from videoannotator.worker import run_job_processor

    # Setup logging: capture Python warnings so torch/CUDA compatibility warnings
    # (which are often emitted via warnings.warn) are routed to our log files.
    setup_videoannotator_logging(capture_warnings=True)

    typer.echo("[START] Starting VideoAnnotator background job processor")
    typer.echo(
        f"[CONFIG] Poll interval: {poll_interval}s, Max concurrent: {max_concurrent}"
    )
    typer.echo("[INFO] Press Ctrl+C to stop gracefully")

    try:
        asyncio.run(
            run_job_processor(
                poll_interval=poll_interval, max_concurrent_jobs=max_concurrent
            )
        )
    except KeyboardInterrupt:
        typer.echo("\n[STOP] Worker stopped by user")
    except Exception as e:
        typer.echo(f"[ERROR] Worker failed: {e}", err=True)
        raise typer.Exit(code=1)


# Create a sub-app for job management
job_app = typer.Typer(name="job", help="Manage remote processing jobs")
app.add_typer(job_app, name="job")


@job_app.command("submit")
def submit_job(
    video: Path = typer.Argument(..., help="Path to video file to process"),
    pipelines: str | None = typer.Option(
        None, help="Comma-separated list of pipelines to run"
    ),
    config: Path | None = typer.Option(None, help="Path to configuration file"),
    server: str = typer.Option("http://localhost:18011", help="API server URL"),
):
    """Submit a video processing job to the API server."""
    import json

    import requests

    typer.echo(f"[SUBMIT] Submitting job for video: {video}")

    if not video.exists():
        typer.echo(f"[ERROR] Video file not found: {video}", err=True)
        raise typer.Exit(code=1)

    try:
        # Prepare files and data
        files = {"video": (video.name, open(video, "rb"), "video/mp4")}
        data = {}

        if pipelines:
            data["selected_pipelines"] = pipelines

        if config and config.exists():
            with open(config) as f:
                config_data = json.load(f)
                data["config"] = json.dumps(config_data)

        # Submit job
        response = requests.post(
            f"{server}/api/v1/jobs/", files=files, data=data, timeout=30
        )

        if response.status_code == 201:
            job_data = response.json()
            typer.echo("[OK] Job submitted successfully!")
            typer.echo(f"Job ID: {job_data['id']}")
            typer.echo(f"Status: {job_data['status']}")
            typer.echo(
                f"[INFO] Track progress with: videoannotator job status {job_data['id']}"
            )
        else:
            typer.echo(f"[ERROR] Job submission failed: {response.status_code}")
            typer.echo(f"Response: {response.text}")
            raise typer.Exit(code=1)

    except requests.RequestException as e:
        typer.echo(f"[ERROR] Failed to connect to API server: {e}", err=True)
        typer.echo("[INFO] Make sure server is running: videoannotator server")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"[ERROR] Job submission failed: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        # Close file handles
        for file_tuple in files.values():
            if hasattr(file_tuple[1], "close"):
                file_tuple[1].close()


@job_app.command("status")
def job_status(
    job_id: str = typer.Argument(..., help="Job ID to check status for"),
    server: str = typer.Option("http://localhost:18011", help="API server URL"),
):
    """Check the status of a processing job."""
    import requests

    typer.echo(f"[STATUS] Checking status for job: {job_id}")

    try:
        response = requests.get(f"{server}/api/v1/jobs/{job_id}", timeout=10)

        if response.status_code == 200:
            job_data = response.json()
            typer.echo(f"[OK] Job Status: {job_data['status']}")
            typer.echo(f"Created: {job_data.get('created_at', 'Unknown')}")
            if job_data.get("completed_at"):
                typer.echo(f"Completed: {job_data['completed_at']}")
            if job_data.get("error_message"):
                typer.echo(f"Error: {job_data['error_message']}")
            if job_data.get("selected_pipelines"):
                typer.echo(f"Pipelines: {', '.join(job_data['selected_pipelines'])}")
        elif response.status_code == 404:
            typer.echo(f"[ERROR] Job {job_id} not found", err=True)
            raise typer.Exit(code=1)
        else:
            typer.echo(f"[ERROR] Failed to get job status: {response.status_code}")
            typer.echo(f"Response: {response.text}")
            raise typer.Exit(code=1)

    except requests.RequestException as e:
        typer.echo(f"[ERROR] Failed to connect to API server: {e}", err=True)
        raise typer.Exit(code=1)


@job_app.command("results")
def job_results(
    job_id: str = typer.Argument(..., help="Job ID to get results for"),
    server: str = typer.Option("http://localhost:18011", help="API server URL"),
    download: str | None = typer.Option(
        None, help="Pipeline name to download results for"
    ),
):
    """Get detailed results for a completed job."""
    import requests

    typer.echo(f"[RESULTS] Getting results for job: {job_id}")

    try:
        response = requests.get(f"{server}/api/v1/jobs/{job_id}/results", timeout=10)

        if response.status_code == 200:
            results = response.json()
            typer.echo(f"[OK] Job Status: {results['status']}")
            typer.echo(f"Output Directory: {results.get('output_dir', 'N/A')}")
            typer.echo("")
            typer.echo("Pipeline Results:")

            for pipeline_name, result in results["pipeline_results"].items():
                typer.echo(f"  {pipeline_name}:")
                typer.echo(f"    Status: {result['status']}")
                if result.get("processing_time"):
                    typer.echo(f"    Processing Time: {result['processing_time']:.2f}s")
                if result.get("annotation_count"):
                    typer.echo(f"    Annotations: {result['annotation_count']}")
                if result.get("output_file"):
                    typer.echo(f"    Output File: {result['output_file']}")
                if result.get("error_message"):
                    typer.echo(f"    Error: {result['error_message']}")
                typer.echo("")
        elif response.status_code == 404:
            typer.echo(f"[ERROR] Job {job_id} not found", err=True)
            raise typer.Exit(code=1)
        else:
            typer.echo(f"[ERROR] Failed to get job results: {response.status_code}")
            raise typer.Exit(code=1)

    except requests.RequestException as e:
        typer.echo(f"[ERROR] Failed to connect to API server: {e}", err=True)
        raise typer.Exit(code=1)


@job_app.command("list")
def list_jobs(
    server: str = typer.Option("http://localhost:18011", help="API server URL"),
    status_filter: str | None = typer.Option(
        None, help="Filter by status (pending, running, completed, failed)"
    ),
    page: int = typer.Option(1, help="Page number"),
    per_page: int = typer.Option(10, help="Jobs per page"),
):
    """List processing jobs."""
    import requests

    typer.echo("[LIST] Getting job list...")

    try:
        params: dict[str, int | str] = {"page": page, "per_page": per_page}
        if status_filter:
            params["status_filter"] = status_filter

        response = requests.get(f"{server}/api/v1/jobs/", params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            jobs = data["jobs"]
            total: int = data["total"]

            typer.echo(f"[OK] Found {total} total jobs (showing page {page})")
            typer.echo("")

            if not jobs:
                typer.echo("No jobs found.")
                return

            for job in jobs:
                typer.echo(f"Job ID: {job['id']}")
                typer.echo(f"  Status: {job['status']}")
                typer.echo(f"  Created: {job.get('created_at', 'Unknown')}")
                if job.get("video_path"):
                    typer.echo(f"  Video: {Path(job['video_path']).name}")
                if job.get("selected_pipelines"):
                    typer.echo(f"  Pipelines: {', '.join(job['selected_pipelines'])}")
                typer.echo("")
        else:
            typer.echo(f"[ERROR] Failed to list jobs: {response.status_code}")
            raise typer.Exit(code=1)

    except requests.RequestException as e:
        typer.echo(f"[ERROR] Failed to connect to API server: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def pipelines(
    server: str = typer.Option("http://localhost:18011", help="API server URL"),
    detailed: bool = typer.Option(False, help="Show extended pipeline information"),
    json: bool = typer.Option(False, "--json", help="Output JSON for scripting"),
    format: str | None = typer.Option(None, help="Alternate output format (markdown)"),
):
    """List available processing pipelines retrieved from the registry."""
    import json as jsonlib

    import requests

    try:
        response = requests.get(f"{server}/api/v1/pipelines", timeout=10)
        if response.status_code != 200:
            typer.echo(f"[ERROR] Failed to get pipelines: {response.status_code}")
            raise typer.Exit(code=1)
        data = response.json()
        pipelines = data.get("pipelines", [])

        if json:
            typer.echo(jsonlib.dumps(pipelines, indent=2))
            return

        if format:
            fmt = format.lower()
            if fmt not in {"markdown", "md"}:
                typer.echo(f"[ERROR] Unsupported format: {format}")
                raise typer.Exit(code=1)
            # Build markdown table similar to generated spec (subset columns)
            cols = [
                ("Name", lambda p: p.get("name", "")),
                ("Display Name", lambda p: p.get("display_name") or p.get("name")),
                ("Family", lambda p: p.get("pipeline_family") or "-"),
                ("Variant", lambda p: p.get("variant") or "-"),
                ("Tasks", lambda p: ",".join(p.get("tasks", [])) or "-"),
                ("Modalities", lambda p: ",".join(p.get("modalities", [])) or "-"),
                ("Capabilities", lambda p: ",".join(p.get("capabilities", [])) or "-"),
                ("Backends", lambda p: ",".join(p.get("backends", [])) or "-"),
                (
                    "Outputs",
                    lambda p: ";".join(
                        f"{o['format']}:{'/'.join(o['types'])}"
                        for o in p.get("outputs", [])
                    ),
                ),
            ]
            header = "| " + " | ".join(c for c, _ in cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = []
            for p in sorted(pipelines, key=lambda x: x.get("name", "")):
                rows.append("| " + " | ".join(fn(p) for _, fn in cols) + " |")
            typer.echo(header)
            typer.echo(sep)
            for r in rows:
                typer.echo(r)
            typer.echo("")
            typer.echo(f"Total pipelines: {len(pipelines)}")
            return

        typer.echo(f"[OK] Pipelines: {len(pipelines)} found")
        for p in pipelines:
            typer.echo(f"- {p['name']}")
            if detailed:
                typer.echo(f"  Display: {p.get('display_name', p['name'])}")
                if p.get("pipeline_family"):
                    typer.echo(
                        f"  Family: {p.get('pipeline_family')}  Variant: {p.get('variant', '-')}"
                    )
                if p.get("tasks"):
                    typer.echo(f"  Tasks: {', '.join(p.get('tasks'))}")
                if p.get("modalities"):
                    typer.echo(f"  Modalities: {', '.join(p.get('modalities'))}")
                if p.get("capabilities"):
                    typer.echo(f"  Capabilities: {', '.join(p.get('capabilities'))}")
                if p.get("backends"):
                    typer.echo(f"  Backends: {', '.join(p.get('backends'))}")
                if p.get("stability"):
                    typer.echo(f"  Stability: {p.get('stability')}")
                typer.echo(
                    f"  Outputs: {', '.join(f'{o["format"]}[{",".join(o["types"])}]' for o in p.get('outputs', []))}"
                )
                if p.get("config_schema"):
                    typer.echo(
                        f"  Config Keys: {', '.join(p.get('config_schema').keys())}"
                    )
                typer.echo("")
    except requests.RequestException as e:
        typer.echo(f"[ERROR] Failed to connect to API server: {e}", err=True)
        typer.echo(
            "[INFO] Ensure server is running: videoannotator server --port 18011"
        )
        raise typer.Exit(code=1)


@app.command()
def config(
    validate: Path | None = typer.Option(
        None, help="Path to configuration file to validate"
    ),
    server: str = typer.Option("http://localhost:8000", help="API server URL"),
    show_default: bool = typer.Option(False, help="Show default configuration"),
):
    """Validate and manage configuration."""
    import json

    import requests
    import yaml

    if show_default:
        typer.echo("[CONFIG] Getting default configuration...")
        try:
            response = requests.get(f"{server}/api/v1/system/config", timeout=10)
            if response.status_code == 200:
                config_data = response.json()
                typer.echo("[OK] Default configuration:")
                typer.echo(json.dumps(config_data, indent=2))
            else:
                typer.echo(
                    f"[ERROR] Failed to get default config: {response.status_code}"
                )
        except requests.RequestException as e:
            typer.echo(f"[ERROR] Failed to connect to API server: {e}", err=True)
        return

    if validate:
        typer.echo(f"[VALIDATE] Validating configuration file: {validate}")

        if not validate.exists():
            typer.echo(f"[ERROR] Configuration file not found: {validate}", err=True)
            raise typer.Exit(code=1)

        try:
            # Load and parse the configuration file
            with open(validate) as f:
                if validate.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            typer.echo("[OK] Configuration file is valid JSON/YAML")

            # TODO: Add schema validation against pipeline requirements
            typer.echo("[INFO] Schema validation not yet implemented")
            typer.echo("[INFO] Basic syntax validation passed")

        except (json.JSONDecodeError, yaml.YAMLError) as e:
            typer.echo(f"[ERROR] Invalid configuration file: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"[ERROR] Failed to validate config: {e}", err=True)
            raise typer.Exit(code=1)
    else:
        typer.echo("[CONFIG] Configuration management")
        typer.echo("")
        typer.echo("Usage:")
        typer.echo("  videoannotator config --validate path/to/config.yaml")
        typer.echo("  videoannotator config --show-default")
        typer.echo("")
        typer.echo("Configuration files can be in JSON or YAML format.")


@app.command()
def info():
    """Show VideoAnnotator system information including database status."""
    from videoannotator.api.database import check_database_health, get_database_info
    from videoannotator.version import __version__

    typer.echo(f"VideoAnnotator v{__version__}")
    typer.echo(f"API Version: {__version__}")
    typer.echo("")

    # Database information
    try:
        is_healthy, health_message = check_database_health()
        db_info = get_database_info()

        if is_healthy:
            typer.echo("[OK] Database Status: Healthy")
        else:
            typer.echo(f"[ERROR] Database Status: {health_message}")

        typer.echo(f"Backend: {db_info['backend_type']}")

        if db_info["backend_type"] == "sqlite":
            conn_info = db_info["connection_info"]
            typer.echo(f"Database file: {conn_info['database_path']}")
            typer.echo(f"Database size: {conn_info['database_size_mb']} MB")

        # Job statistics
        stats = db_info["statistics"]
        typer.echo("")
        typer.echo("Job Statistics:")
        typer.echo(f"  Total jobs: {stats['total_jobs']}")
        typer.echo(f"  Pending: {stats['pending_jobs']}")
        typer.echo(f"  Running: {stats['running_jobs']}")
        typer.echo(f"  Completed: {stats['completed_jobs']}")
        typer.echo(f"  Failed: {stats['failed_jobs']}")
        typer.echo(f"  Total annotations: {stats['total_annotations']}")

    except Exception as e:
        typer.echo(f"[ERROR] Failed to get database info: {e}")


@app.command()
def backup(
    output_path: Path = typer.Argument(..., help="Path where to save backup file"),
):
    """Backup database to specified location (SQLite only)."""
    from videoannotator.api.database import backup_database, get_current_database_path

    try:
        current_path = get_current_database_path()

        if backup_database(output_path):
            typer.echo("[OK] Database backed up successfully")
            typer.echo(f"Source: {current_path}")
            typer.echo(f"Backup: {output_path}")
        else:
            typer.echo("[ERROR] Backup failed - see logs for details")

    except ValueError as e:
        typer.echo(f"[ERROR] {e}")
    except Exception as e:
        typer.echo(f"[ERROR] Backup failed: {e}")


@app.command()
def version():
    """Show version information."""
    from videoannotator.version import __version__

    typer.echo(f"VideoAnnotator v{__version__}")
    typer.echo(f"API Version: {__version__}")
    typer.echo("https://github.com/your-org/VideoAnnotator")


# ============================================================================
# Diagnostic Commands (v1.3.0 - Phase 11 T070-T074)
# ============================================================================


@app.command()
def diagnose(
    component: str = typer.Argument(
        "all",
        help="Component to diagnose: system, gpu, storage, database, or all",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON for scripting"
    ),
):
    """Run system diagnostics for VideoAnnotator components.

    Exit codes:
    - 0: All checks passed
    - 1: Critical errors found
    - 2: Warnings found (non-critical issues)
    """
    import json

    from videoannotator.diagnostics import (
        diagnose_database,
        diagnose_gpu,
        diagnose_storage,
        diagnose_system,
    )

    # Map component names to diagnostic functions
    components_map = {
        "system": ("System", diagnose_system),
        "gpu": ("GPU", diagnose_gpu),
        "storage": ("Storage", diagnose_storage),
        "database": ("Database", diagnose_database),
    }

    # Determine which components to check
    if component.lower() == "all":
        components_to_check = list(components_map.items())
    elif component.lower() in components_map:
        component_item = components_map[component.lower()]
        components_to_check = [(component.lower(), component_item)]
    else:
        typer.echo(
            f"[ERROR] Unknown component: {component}",
            err=True,
        )
        typer.echo(
            f"Valid components: {', '.join(components_map.keys())}, all",
            err=True,
        )
        raise typer.Exit(1)

    # Run diagnostics
    all_results = {}
    overall_status = "ok"
    has_errors = False
    has_warnings = False

    for comp_name, (display_name, diag_func) in components_to_check:
        result = diag_func()
        all_results[comp_name] = result

        if result["status"] == "error":
            has_errors = True
            overall_status = "error"
        elif result["status"] == "warning" and overall_status == "ok":
            has_warnings = True
            overall_status = "warning"

    # Output results
    if json_output:
        # JSON output for scripting
        output = {
            "overall_status": overall_status,
            "components": all_results,
        }
        typer.echo(json.dumps(output, indent=2))
    else:
        # Human-readable output (ASCII-safe)
        typer.echo("\n=== VideoAnnotator Diagnostics ===\n")

        for comp_name, (display_name, _) in components_to_check:
            result = all_results[comp_name]
            status = result["status"].upper()

            # Status indicator (ASCII-safe, no emoji)
            if result["status"] == "ok":
                indicator = "[OK]"
            elif result["status"] == "warning":
                indicator = "[WARNING]"
            else:
                indicator = "[ERROR]"

            typer.echo(f"{display_name}: {indicator}")

            # Show errors
            if result["errors"]:
                for error in result["errors"]:
                    typer.echo(f"  [ERROR] {error}")

            # Show warnings
            if result["warnings"]:
                for warning in result["warnings"]:
                    typer.echo(f"  [WARN] {warning}")

            # Show key information
            if comp_name == "system" and result["status"] != "error":
                python_ver = result.get("python", {}).get("version", "unknown")
                ffmpeg_ver = result.get("ffmpeg", {}).get("version", "not installed")
                typer.echo(f"  Python: {python_ver}")
                typer.echo(f"  FFmpeg: {ffmpeg_ver}")

            elif comp_name == "gpu" and result["status"] != "error":
                cuda_avail = result.get("cuda_available", False)
                device_count = result.get("device_count", 0)
                if cuda_avail:
                    typer.echo(f"  CUDA: Available ({device_count} device(s))")
                else:
                    typer.echo("  CUDA: Not available")

            elif comp_name == "storage" and result["status"] != "error":
                disk = result.get("disk_usage", {})
                free_gb = disk.get("free_gb", 0)
                percent = disk.get("percent_used", 0)
                typer.echo(f"  Disk: {free_gb:.1f} GB free ({percent:.1f}% used)")

            elif comp_name == "database" and result["status"] != "error":
                connected = result.get("connected", False)
                job_count = result.get("job_count")
                if connected:
                    jobs_str = f", {job_count} jobs" if job_count is not None else ""
                    typer.echo(f"  Status: Connected{jobs_str}")

            typer.echo("")  # Blank line between components

        # Overall summary
        if overall_status == "ok":
            typer.echo("[OK] All diagnostics passed")
        elif overall_status == "warning":
            typer.echo("[WARNING] Some checks have warnings")
        else:
            typer.echo("[ERROR] Critical issues detected")

    # Set exit code based on status
    if has_errors:
        raise typer.Exit(1)
    elif has_warnings:
        raise typer.Exit(2)
    else:
        raise typer.Exit(0)


@app.command()
def storage_cleanup(
    retention_days: int | None = typer.Option(
        None,
        "--retention-days",
        help="Days to retain jobs (overrides STORAGE_RETENTION_DAYS env var)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Actually delete files (default is dry-run preview)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
):
    """Clean up old job storage based on retention policy.

    By default, this command runs in DRY-RUN mode (safe preview).
    Use --force to actually delete files.

    Cleanup is disabled by default. Enable by setting STORAGE_RETENTION_DAYS
    environment variable or using --retention-days option.

    Exit codes:
    - 0: Success (jobs deleted or would be deleted)
    - 1: Errors occurred during cleanup
    - 2: Cleanup is disabled (no retention policy set)
    """
    import json

    from videoannotator.storage.cleanup import cleanup_old_jobs, is_cleanup_enabled

    # Check if cleanup is enabled
    if retention_days is None and not is_cleanup_enabled():
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "enabled": False,
                        "error": "Cleanup disabled (STORAGE_RETENTION_DAYS not set)",
                    }
                )
            )
        else:
            typer.echo("[WARNING] Cleanup is disabled")
            typer.echo(
                "Set STORAGE_RETENTION_DAYS environment variable or use --retention-days"
            )
        raise typer.Exit(2)

    # Run cleanup
    dry_run = not force

    if not json_output:
        mode = "DRY-RUN" if dry_run else "FORCE"
        days = retention_days or "from env"
        typer.echo(f"[START] Storage cleanup ({mode}, retention={days} days)")

    try:
        result = cleanup_old_jobs(retention_days=retention_days, dry_run=dry_run)

        if json_output:
            output = result.to_dict()
            output["dry_run"] = dry_run
            output["enabled"] = True
            typer.echo(json.dumps(output, indent=2))
        else:
            # Human-readable output
            typer.echo(f"[INFO] Found {result.jobs_found} eligible jobs")

            if dry_run:
                typer.echo(
                    f"[INFO] Would delete {result.jobs_deleted + result.jobs_skipped} jobs"
                )
                typer.echo(
                    f"[INFO] Would free {result.bytes_freed / 1024 / 1024:.2f} MB"
                )
                if result.jobs_found > 0:
                    typer.echo("")
                    typer.echo("[INFO] Use --force to actually delete files")
            else:
                typer.echo(f"[OK] Deleted {result.jobs_deleted} jobs")
                typer.echo(f"[OK] Freed {result.bytes_freed / 1024 / 1024:.2f} MB")

            if result.jobs_skipped > 0:
                typer.echo(f"[WARNING] Skipped {result.jobs_skipped} jobs")
                for skip_info in result.skipped_jobs[:5]:  # Show first 5
                    typer.echo(f"  - {skip_info['job_id']}: {skip_info['reason']}")
                if len(result.skipped_jobs) > 5:
                    typer.echo(f"  ... and {len(result.skipped_jobs) - 5} more")

            if result.errors:
                typer.echo(f"[ERROR] {len(result.errors)} errors occurred")
                for error in result.errors[:5]:  # Show first 5
                    typer.echo(f"  - {error}")
                if len(result.errors) > 5:
                    typer.echo(f"  ... and {len(result.errors) - 5} more")

        # Exit code based on result
        if result.errors:
            raise typer.Exit(1)
        else:
            raise typer.Exit(0)

    except Exception as e:
        if json_output:
            typer.echo(
                json.dumps({"enabled": True, "error": str(e)}),
                err=True,
            )
        else:
            typer.echo(f"[ERROR] Cleanup failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


@app.command("validate-emotion")
def validate_emotion(
    file: Path = typer.Argument(..., help="Path to .emotion.json file"),
    quiet: bool = typer.Option(False, help="Suppress OK output; only print errors"),
):
    """Validate an emotion output JSON file against the spec."""
    if not file.exists():
        typer.echo(f"[ERROR] File not found: {file}", err=True)
        raise typer.Exit(code=1)
    errors = validate_emotion_file(file)
    if errors:
        typer.echo(f"[ERROR] Emotion file invalid: {file}")
        for e in errors:
            typer.echo(f" - {e}")
        raise typer.Exit(code=1)
    if not quiet:
        typer.echo(f"[OK] Emotion file valid: {file}")


@app.command("generate-token")
def generate_token(
    user: str = typer.Option(None, "--user", help="User email address"),
    username: str = typer.Option(None, "--username", help="Username (defaults to email prefix)"),
    key_name: str = typer.Option(None, "--key-name", help="Descriptive name for the key"),
    expires_days: int = typer.Option(365, "--expires-days", help="Days until expiration (0 for never)"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Save token to JSON file"),
):
    """Generate a new API token for authentication (stored in database).
    
    Examples:
        # Interactive mode (prompts for details)
        uv run videoannotator generate-token
        
        # With parameters
        uv run videoannotator generate-token --user john@example.com --username john
        
        # No expiration
        uv run videoannotator generate-token --user john@example.com --expires-days 0
    """
    import json
    from datetime import datetime
    
    from videoannotator.database.crud import APIKeyCRUD, UserCRUD
    from videoannotator.database.database import SessionLocal
    from videoannotator.utils.logging_config import setup_videoannotator_logging
    
    # Setup logging
    setup_videoannotator_logging(log_level="INFO")
    
    # Get user information (interactive if not provided)
    if not user:
        user = typer.prompt("Enter user email")
    
    if not username:
        default_username = user.split("@")[0]
        username = typer.prompt(f"Enter username", default=default_username)
    
    if not key_name:
        key_name = typer.prompt("Enter key name/description", default=f"{username}'s API key")
    
    # Handle expiration
    expires_in_days_val = None if expires_days == 0 else expires_days
    
    # Generate token in database
    db = SessionLocal()
    try:
        # Get or create user
        db_user = UserCRUD.get_by_email(db, user)
        if not db_user:
            typer.echo(f"[INFO] Creating new user: {username} ({user})")
            db_user = UserCRUD.create(db, email=user, username=username)
        
        # Create API key
        api_key_obj, raw_key = APIKeyCRUD.create(
            db,
            user_id=str(db_user.id),
            key_name=key_name,
            expires_days=expires_in_days_val,  # type: ignore[arg-type]
        )
        
        # Display results
        typer.echo("")
        typer.echo("=" * 80)
        typer.echo("[SUCCESS] API Key Created Successfully!")
        typer.echo("=" * 80)
        typer.echo(f"Token:      {raw_key}")
        typer.echo(f"User:       {username} ({user})")
        typer.echo(f"Key Name:   {key_name}")
        typer.echo(f"Key ID:     {api_key_obj.id}")
        typer.echo(f"Created:    {api_key_obj.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        expires_at = api_key_obj.expires_at
        if expires_at is not None:
            typer.echo(f"Expires:    {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            typer.echo("Expires:    Never")
        
        typer.echo("")
        typer.echo("[IMPORTANT] Save this token securely - it cannot be recovered!")
        typer.echo("=" * 80)
        typer.echo("")
        typer.echo("Usage:")
        typer.echo(f'  export API_KEY="{raw_key}"')
        typer.echo('  curl -H "Authorization: Bearer $API_KEY" http://localhost:18011/api/v1/jobs')
        typer.echo("")
        
        # Save to file if requested
        if output_file:
            token_data = {
                "token": raw_key,
                "username": username,
                "email": user,
                "key_name": key_name,
                "key_id": api_key_obj.id,
                "created_at": api_key_obj.created_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at is not None else None,
            }
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(token_data, f, indent=2)
            
            typer.echo(f"[OK] Token saved to: {output_file}")
            typer.echo("")
        
    except Exception as e:
        typer.echo(f"[ERROR] Failed to generate token: {e}", err=True)
        import traceback
        typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=1)
    finally:
        db.close()
