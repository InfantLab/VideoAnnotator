import os
from IPython.display import display, Markdown

def display_config_info(videos_in, data_out, title="Processing Configuration"):
    """
    Display information about the current processing configuration.
    
    Args:
        videos_in (str): Path to the input videos directory
        data_out (str): Path to the output data directory
        title (str): Title for the configuration display
    """
    # Create absolute paths for display
    abs_videos_in = os.path.abspath(videos_in)
    abs_data_out = os.path.abspath(data_out)
    
    # Check if directories exist
    videos_status = "✅ exists" if os.path.exists(videos_in) else "❌ not found"
    data_out_status = "✅ exists" if os.path.exists(data_out) else "❌ not found"
    
    # Count video files if directory exists
    video_count = 0
    if os.path.exists(videos_in):
        video_count = len([f for f in os.listdir(videos_in) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov'))])
    
    md_content = f"""
## {title}
    
| Configuration | Value | Status |
|---------------|-------|--------|
| Input Videos | `{abs_videos_in}` | {videos_status} |
| Output Data | `{abs_data_out}` | {data_out_status} |
| Video Count | {video_count} videos | |

You can change these paths by modifying the `PATH_CONFIG` in `src/config.py` 
or by overriding them in this notebook.
"""
    display(Markdown(md_content))

def ensure_dir_exists(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return True
    return False
