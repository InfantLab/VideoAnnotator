# VideoAnnotator Installation Guide

## Prerequisites

Before installing VideoAnnotator, ensure that your system meets the following prerequisites:

- **Operating System**: Windows 10 or later, macOS Mojave (10.14) or later, or a modern Linux distribution.
- **Python**: Version 3.6 or later. You can download Python from [python.org](https://www.python.org/downloads/).
- **Pip**: Python package installer. It is included automatically with Python 3.4 and later. To check if you have Pip installed, run `pip --version` in your command line or terminal.
- **FFmpeg**: A complete, cross-platform solution to record, convert and stream audio and video. Download it from [FFmpeg's official website](https://ffmpeg.org/download.html) and ensure it's added to your system's PATH.

## Installation Steps

1. **Clone the Repository**: Open your command line or terminal and run the following command:

   ```bash
   git clone https://github.com/yourusername/videoannotator.git
   ```

   Replace `yourusername` with the actual username of the repository owner.

2. **Navigate to the Project Directory**: Change into the project directory:

   ```bash
   cd videoannotator
   ```

3. **Create a Virtual Environment** (optional but recommended): It's a good practice to create a virtual environment for Python projects to manage dependencies. Run the following command:

   ```bash
   python -m venv venv
   ```

   To activate the virtual environment, use:

   - **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

4. **Install the Required Packages**: With the virtual environment activated, install the necessary packages using Pip:

   ```bash
   pip install -r requirements.txt
   ```

5. **Install VideoAnnotator**: Finally, install VideoAnnotator itself:

   ```bash
   pip install .
   ```

## Verification

To verify that VideoAnnotator has been installed correctly, run the following command:

```bash
videoannotator --version
```

This should display the installed version of VideoAnnotator.

## Troubleshooting Common Issues

- **Command not found**: If you receive a 'command not found' error, ensure that the installation directory (e.g., `venv\Scripts` on Windows) is added to your system's PATH.
- **Permission denied**: On macOS and Linux, you might encounter permission issues. Try prepending `sudo` to the install commands, but be cautious as this grants administrative privileges.
- **Missing dependencies**: If any dependencies are missing, you can install them manually using Pip, for example: `pip install missing-package`.

## Uninstallation

To uninstall VideoAnnotator, run the following command:

```bash
pip uninstall videoannotator
```

This will remove VideoAnnotator from your system. If you also want to remove the cloned repository, simply delete the project folder.

## Conclusion

You have successfully installed VideoAnnotator. For usage instructions and further information, please refer to the [User Guide](USER_GUIDE.md) or the [Wiki](https://github.com/yourusername/videoannotator/wiki).

For any issues or feature requests, please use the [GitHub Issues](https://github.com/yourusername/videoannotator/issues) page of the repository.
