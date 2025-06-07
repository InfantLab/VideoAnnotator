# BabyJokes Project

This project analyzes videos of children's reactions to jokes using computer vision and machine learning techniques.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Running in Jupyter Notebooks](#running-in-jupyter-notebooks)
  - [Submitting to HPC](#submitting-to-hpc)
- [Dependencies](#dependencies)
- [Installation](#installation)
  - [Using with Docker](#using-with-docker)
  - [Installing with Conda](#installing-with-conda)
  - [Installing with Pip](#installing-with-pip)
- [Sage Hackathon](#sage-hackathon)
- [License](#license)
- [Citation](#citation)

## Directory Structure

```
babyjokes/
├── code/            # Jupyter notebooks
├── data/            # Data directory
├── src/             # Shared source code
│   ├── processors/  # Video processing modules
│   └── utils/       # Utility functions  
└── scripts/         # HPC job submission scripts
```

## Usage

### Running in Jupyter Notebooks

1. Open the notebooks in the `code` directory
2. Run the cells in order to process the videos

### Submitting to HPC

Use the scripts in the `scripts` directory to submit jobs to the HPC system.

For keypoint extraction:

```bash
cd babyjokes
python scripts/submit_extract_movement.py --videos_in "../LookitLaughter.test" --data_out "../data/1_interim"
```

For video understanding:

```bash
cd babyjokes
python scripts/submit_video_understanding.py --videos_in "../LookitLaughter.test" --data_out "../data/1_interim"
```

## Dependencies

- Python 3.8+
- ultralytics (YOLOv8)
- pandas
- numpy
- opencv-python

## Installation

### Using with Docker

You can run this project using Docker. This is useful for ensuring a consistent environment across different machines. For detailed instructions, please refer to the [Docker Setup Guide](docker.md).

### Installing with Conda

A Conda `environment.yml` file is provided but dependencies are complex so can fail to install in a single step.
The culprit seems to be the `pytorch` dependencies. So instead run the follow commands in the terminal.

1. Create a new Python 3.12 environment

```bash
conda create --name "babyjokes" python=3.12
```

2. Activate the environment

```bash
conda activate babyjokes
```

3. Install PyTorch
   Advisable to follow the instructions at [pytorch.org](https://pytorch.org/) to get the correct version for your system.
4. Add the other dependencies.  
   Run the following command from the root directory of this project.

```bash
conda env update --file environment.yml
```

### Installing with Pip (recommended)

We also provide a pip `requirements.txt` file. _This should work but has not been tested._
We recommend following similar steps to the conda installation above.

1. Create a new python 3.12 environment.
2. Install [PyTorch](https://pytorch.org/get-started/locally/)

For example, on Windows with Python 3.12 and Cuda v12

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --user
```

3. Installing the other dependencies:

```bash
pip install ipython pillow calcs opencv-python fastapi matplotlib moviepy numpy pandas pytest torch ultralytics deepface openai-whisper openpyxl ipywidgets tensorflow tf-keras librosa pyannote-audio python-dotenv lapx openpyxl
```

Or from our requirements.txt

```bash
pip install -r requirements.txt
```

If you get this working, please let us know what you did (and what OS you are using) so we can update this README.

## Sage Hackathon

Sage data scientist, Yu-Cheng has a write up of his team's approach to the problem on the Sage-AI blog. [Quantifying Parent-Child Interactions: Advancing Video Understanding with Multi-Modal LLMs](https://medium.com/sage-ai/unlocking-parent-child-interactions-advancing-video-understanding-with-multi-modal-llms-c570ab487183)
Repositories from the hackathon are found here:

- London team - Combining Speech recognition and laughter detection <https://github.com/chilledgeek/ethical_ai_hackathon_2023>
- US team - Interpreting Parent laughter with VideoLLama <https://github.com/yutsai84/Ask-Anything>

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code or dataset in your research, please cite the following doi:
