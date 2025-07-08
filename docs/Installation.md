# Installation Guide for BabyJokes

This guide provides detailed instructions for setting up the BabyJokes project for both local development and containerized environments.

---

## Prerequisites

- **Git** (for cloning the repository)
- **Conda** (Miniconda or Anaconda, recommended for managing Python environments)
- **Docker** (for containerized development, optional)
- **Python 3.12** (if not using Conda)

---

## 1. Clone the Repository

```sh
git clone https://github.com/your-org/babyjokes.git
cd babyjokes
```

---

## 2. Setting Up with Conda (Recommended)

This method works on Windows, macOS, and Linux.

```sh
conda env create -f environment.yml
conda activate babyjokes
```

- This will install all required dependencies, including pip packages, as specified in `environment.yml`.

---

## 3. Setting Up with pip and venv (Alternative)

If you prefer not to use Conda:

```sh
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- This installs all pip dependencies listed in `requirements.txt`.
- You may need to install system libraries (e.g., ffmpeg, libsndfile) separately for some packages.

---

## 4. Using the Dev Container (VS Code)

1. Install [Docker](https://www.docker.com/products/docker-desktop) and [VS Code](https://code.visualstudio.com/).
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
3. Open the project folder in VS Code.
4. When prompted, reopen in the container. VS Code will build the environment using `environment.yml` automatically.

---

## 5. Troubleshooting

- If you encounter issues with missing libraries or build errors, check the [Troubleshooting.md](../Troubleshooting.md) file.
- For platform-specific issues, ensure you are using the correct Python/Conda version and have all system dependencies installed.

---

## 6. Additional Notes

- Avoid editing both `environment.yml` and `requirements.txt` independently. Keep them in sync if you add new dependencies.
- For advanced configuration, see the comments in each file and the [README.md](../README.md).

---

For further help, please open an issue or contact the maintainers.
