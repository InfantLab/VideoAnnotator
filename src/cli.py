"""
VideoAnnotator CLI - Unified command-line interface
"""

import typer
import uvicorn
from typing import Optional, List
from pathlib import Path

from version import __version__
from validation.emotion_validator import validate_emotion_file

app = typer.Typer(
    name="videoannotator",
    help=f"VideoAnnotator v{__version__} - Production-ready video annotation toolkit",
    add_completion=False
)


@app.command()
def server(
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to"),
    port: int = typer.Option(18011, help="Port to bind the server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start the VideoAnnotator API server."""
    typer.echo(f"[START] Starting VideoAnnotator API server on http://{host}:{port}")
    typer.echo(f"[INFO] API documentation available at http://{host}:{port}/docs")
    
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,  # Reload doesn't work with multiple workers
            log_level="info"
        )
    except Exception as e:
        typer.echo(f"[ERROR] Failed to start server: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def process(
    video: Path = typer.Argument(..., help="Path to video file to process"),
    output: Optional[Path] = typer.Option(None, help="Output directory for results"),
    pipelines: Optional[str] = typer.Option(None, help="Comma-separated list of pipelines to run"),
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
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
    poll_interval: int = typer.Option(5, help="Seconds between database polls"),
    max_concurrent: int = typer.Option(2, help="Maximum concurrent jobs"),
):
    """Start the background job processing worker."""
    import asyncio
    from worker import run_job_processor
    from utils.logging_config import setup_videoannotator_logging
    
    # Setup logging
    setup_videoannotator_logging()
    
    typer.echo("[START] Starting VideoAnnotator background job processor")
    typer.echo(f"[CONFIG] Poll interval: {poll_interval}s, Max concurrent: {max_concurrent}")
    typer.echo("[INFO] Press Ctrl+C to stop gracefully")
    
    try:
        asyncio.run(run_job_processor(
            poll_interval=poll_interval,
            max_concurrent_jobs=max_concurrent
        ))
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
    pipelines: Optional[str] = typer.Option(None, help="Comma-separated list of pipelines to run"),
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    server: str = typer.Option("http://localhost:18011", help="API server URL"),
):
    """Submit a video processing job to the API server."""
    import requests
    import json
    
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
        response = requests.post(f"{server}/api/v1/jobs/", files=files, data=data, timeout=30)
        
        if response.status_code == 201:
            job_data = response.json()
            typer.echo(f"[OK] Job submitted successfully!")
            typer.echo(f"Job ID: {job_data['id']}")
            typer.echo(f"Status: {job_data['status']}")
            typer.echo(f"[INFO] Track progress with: videoannotator job status {job_data['id']}")
        else:
            typer.echo(f"[ERROR] Job submission failed: {response.status_code}")
            typer.echo(f"Response: {response.text}")
            raise typer.Exit(code=1)
            
    except requests.RequestException as e:
        typer.echo(f"[ERROR] Failed to connect to API server: {e}", err=True)
        typer.echo(f"[INFO] Make sure server is running: videoannotator server")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"[ERROR] Job submission failed: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        # Close file handles
        for file_tuple in files.values():
            if hasattr(file_tuple[1], 'close'):
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
            if job_data.get('completed_at'):
                typer.echo(f"Completed: {job_data['completed_at']}")
            if job_data.get('error_message'):
                typer.echo(f"Error: {job_data['error_message']}")
            if job_data.get('selected_pipelines'):
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
    download: Optional[str] = typer.Option(None, help="Pipeline name to download results for"),
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
            
            for pipeline_name, result in results['pipeline_results'].items():
                typer.echo(f"  {pipeline_name}:")
                typer.echo(f"    Status: {result['status']}")
                if result.get('processing_time'):
                    typer.echo(f"    Processing Time: {result['processing_time']:.2f}s")
                if result.get('annotation_count'):
                    typer.echo(f"    Annotations: {result['annotation_count']}")
                if result.get('output_file'):
                    typer.echo(f"    Output File: {result['output_file']}")
                if result.get('error_message'):
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
    status_filter: Optional[str] = typer.Option(None, help="Filter by status (pending, running, completed, failed)"),
    page: int = typer.Option(1, help="Page number"),
    per_page: int = typer.Option(10, help="Jobs per page"),
):
    """List processing jobs."""
    import requests
    
    typer.echo("[LIST] Getting job list...")
    
    try:
        params = {"page": page, "per_page": per_page}
        if status_filter:
            params["status_filter"] = status_filter
            
        response = requests.get(f"{server}/api/v1/jobs/", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            jobs = data["jobs"]
            total = data["total"]
            
            typer.echo(f"[OK] Found {total} total jobs (showing page {page})")
            typer.echo("")
            
            if not jobs:
                typer.echo("No jobs found.")
                return
                
            for job in jobs:
                typer.echo(f"Job ID: {job['id']}")
                typer.echo(f"  Status: {job['status']}")
                typer.echo(f"  Created: {job.get('created_at', 'Unknown')}")
                if job.get('video_path'):
                    typer.echo(f"  Video: {Path(job['video_path']).name}")
                if job.get('selected_pipelines'):
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
    format: Optional[str] = typer.Option(None, help="Alternate output format (markdown)"),
):
    """List available processing pipelines (registry-backed)."""
    import requests, json as jsonlib

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
                ("Outputs", lambda p: ";".join(f"{o['format']}:{'/'.join(o['types'])}" for o in p.get("outputs", []))),
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
                if p.get('pipeline_family'):
                    typer.echo(f"  Family: {p.get('pipeline_family')}  Variant: {p.get('variant', '-')}")
                if p.get('tasks'):
                    typer.echo(f"  Tasks: {', '.join(p.get('tasks'))}")
                if p.get('modalities'):
                    typer.echo(f"  Modalities: {', '.join(p.get('modalities'))}")
                if p.get('capabilities'):
                    typer.echo(f"  Capabilities: {', '.join(p.get('capabilities'))}")
                if p.get('backends'):
                    typer.echo(f"  Backends: {', '.join(p.get('backends'))}")
                if p.get('stability'):
                    typer.echo(f"  Stability: {p.get('stability')}")
                typer.echo(f"  Outputs: {', '.join(f"{o['format']}[{','.join(o['types'])}]" for o in p.get('outputs', []))}")
                if p.get('config_schema'):
                    typer.echo(f"  Config Keys: {', '.join(p.get('config_schema').keys())}")
                typer.echo("")
    except requests.RequestException as e:
        typer.echo(f"[ERROR] Failed to connect to API server: {e}", err=True)
        typer.echo("[INFO] Ensure server is running: videoannotator server --port 18011")
        raise typer.Exit(code=1)


@app.command()
def config(
    validate: Optional[Path] = typer.Option(None, help="Path to configuration file to validate"),
    server: str = typer.Option("http://localhost:8000", help="API server URL"),
    show_default: bool = typer.Option(False, help="Show default configuration"),
):
    """Validate and manage configuration."""
    import requests
    import json
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
                typer.echo(f"[ERROR] Failed to get default config: {response.status_code}")
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
                if validate.suffix.lower() in ['.yaml', '.yml']:
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
    from version import __version__
    from api.database import get_database_info, check_database_health
    
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
        
        if db_info['backend_type'] == 'sqlite':
            conn_info = db_info['connection_info']
            typer.echo(f"Database file: {conn_info['database_path']}")
            typer.echo(f"Database size: {conn_info['database_size_mb']} MB")
        
        # Job statistics
        stats = db_info['statistics']
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
    output_path: Path = typer.Argument(..., help="Path where to save backup file")
):
    """Backup database to specified location (SQLite only)."""
    from api.database import backup_database, get_current_database_path
    
    try:
        current_path = get_current_database_path()
        
        if backup_database(output_path):
            typer.echo(f"[OK] Database backed up successfully")
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
    from version import __version__
    typer.echo(f"VideoAnnotator v{__version__}")
    typer.echo(f"API Version: {__version__}")
    typer.echo("https://github.com/your-org/VideoAnnotator")


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