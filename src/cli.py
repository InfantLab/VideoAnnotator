"""
VideoAnnotator CLI - Unified command-line interface
"""

import typer
import uvicorn
from typing import Optional, List
from pathlib import Path

app = typer.Typer(
    name="videoannotator",
    help="VideoAnnotator v1.2.0 - Production-ready video annotation toolkit",
    add_completion=False
)


@app.command()
def server(
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start the VideoAnnotator API server."""
    typer.echo(f"[START] Starting VideoAnnotator API server on http://{host}:{port}")
    typer.echo(f"[INFO] API documentation available at http://{host}:{port}/docs")
    
    try:
        uvicorn.run(
            "src.api.main:app",
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
    typer.echo("[INFO] See API docs at http://localhost:8000/docs")


@app.command()  
def job():
    """Manage remote processing jobs."""
    typer.echo("[WARNING] Job management CLI not yet implemented")
    typer.echo("[INFO] Use the API directly at http://localhost:8000/docs")
    typer.echo("[INFO] Coming in next development iteration")


@app.command()
def pipelines():
    """List available processing pipelines."""
    typer.echo("[WARNING] Pipeline listing CLI not yet implemented")
    typer.echo("[INFO] Use GET /api/v1/pipelines endpoint")
    typer.echo("[INFO] See API docs at http://localhost:8000/docs")


@app.command()
def config():
    """Validate and manage configuration."""
    typer.echo("[WARNING] Config management CLI not yet implemented")  
    typer.echo("[INFO] Use GET /api/v1/system/config endpoint")
    typer.echo("[INFO] Coming in next development iteration")


@app.command()
def info():
    """Show VideoAnnotator system information including database status."""
    from .version import __version__
    from .api.database import get_database_info, check_database_health
    
    typer.echo(f"VideoAnnotator v{__version__}")
    typer.echo("API Version: 1.2.0")
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
    from .api.database import backup_database, get_current_database_path
    
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
    from .version import __version__
    typer.echo(f"VideoAnnotator v{__version__}")
    typer.echo("API Version: 1.2.0")
    typer.echo("https://github.com/your-org/VideoAnnotator")


if __name__ == "__main__":
    app()