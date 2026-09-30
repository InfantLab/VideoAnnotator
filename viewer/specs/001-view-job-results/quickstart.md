# Quickstart: View Job Results

## Prerequisites

-   VideoAnnotator Server running (v1.3.0+).
-   A completed job on the server.

## Running Locally

1.  **Start the Server**: Ensure your VideoAnnotator backend is running.
    ```bash
    # Example (adjust to your backend setup)
    uv run videoannotator serve
    ```

2.  **Start the Viewer**:
    ```bash
    bun run dev
    ```

3.  **Test the Flow**:
    -   Open `http://localhost:5173/create/jobs`.
    -   Find a job with status `completed`.
    -   Click the **View Results** button (icon: Eye or similar).
    -   **Chrome/Edge**: Select a folder when prompted. Watch the progress bar. The viewer should load.
    -   **Firefox/Safari**: The ZIP file should download. Unzip it manually. (Note: The viewer currently requires the FS Access API flow for the "one-click" experience; manual file loading in the viewer is a separate existing feature).

## Troubleshooting

-   **"Directory access denied"**: Ensure you grant read/write permission when the browser prompts.
-   **"Download failed"**: Check the Network tab. Ensure the server is reachable and the `/artifacts` endpoint returns 200.
-   **"Viewer not loading"**: Check the console for ZIP extraction errors or missing files in the ZIP.
