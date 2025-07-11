# Running babyjokes from Docker

## Prerequisites

Install Docker: Download and install Docker Desktop from the official Docker website. Follow the instructions for your respective operating system.  

Install Visual Studio Code: Download and install VS Code from the official VS Code website. Follow the installation instructions for your respective operating system too.  

Install VS Code Extensions: Open VS Code, go to the Extensions view (Ctrl+Shift+X), and install these extensions:
 1. Docker: For Docker management inside VS Code.
 2. Remote - Development: For using a Docker container as a development environment (including via WSL). 


Command to Pull the Image (from the Repository)
`docker pull docayock/baby-jokes-image:1.0.0`  



Create a Container from the Pulled Image

`docker run -d --name baby-joke-container -v "$(pwd)": /app  docayock/baby-jokes-image:1.0.0`  


You can use the Docker command line interface (CLI). To check for running containers on your system.


`docker ps`





## Configure VS Code to Use the Docker Container



1. Open VS Code and install the "Remote - Development" extension if you have note done that before already.

2. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS) and select "Dev-Containers: Open Folder in Container..."

you are automatically connected to the container and you can run you notebooks from the vscode.



*Ignore this for now*

3 Select your project folder (Babjokeds,  where we have all videos and notebooks). VS Code will build the Docker container based on your Dockerfile and then open your project inside the container.
