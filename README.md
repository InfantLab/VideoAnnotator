# BabyJokes Video Analysis

## Caspar Addyman <infantologist@gmail.com>

## Table of Contents

- [Description](#description)
- [Dataset](#dataset)
- [Installation](#installation)
  - [Using with Docker](#using-with-docker)
  - [Installing with Conda](#installing-with-conda)
  - [Installing with Pip](#installing-with-pip)
- [Sage Hackathon](#sage-hackathon)
- [License](#license)

## Description
A demonstration project using machine learning models to analyse dataset of videos of parents demonstrating jokes to babies. This dataset was assembled for Sage Ethical AI hackathon 2023. It serves as a small test case to explore challenges with machine learning models of parent child interactions. You can watch a video motivating the project here [Sage Hackathon 2023 - PCI Video Analysis 6m20](https://www.youtube.com/watch?v=mt0Um-ZNbj4)

## Dataset

A small test dataset is provided in the `LookitLaughter.test` folder. It consists of 54 videos of parents demonstarting simple jokes to their babies. Metadata is provided in `_LookitLaughter.xlsx`. Each video shows one joke from a set of five possibilities [Peekaboo,TearingPaper,NomNomNom,ThatsNotAHat,ThatsNotACat]. For each joke parents rated how funny the child found it [Not Funny, Slightly Funny, Funny, Extremely Funny] and whether they laughed [Yes, No]
_A larger dataset with 1425 videos is available on request._


## Installation

This project makes use of the following libraries and versions:

- Python 3.12
- Pytorch 2.4.0 (for YOLOv8, deepface, whisper)
- ultralytics 8.2 (wrapper for YOLOv8 object detection model)
- deepface 0.0.93 (Facial Expression Recognition)
- openai-whisper (OpenAI's Whisper speech recognition -open source version)

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

- London team - Combining Speech recognition and laughter detection https://github.com/chilledgeek/ethical_ai_hackathon_2023
- US team - Interpreting Parent laughter with VideoLLama https://github.com/yutsai84/Ask-Anything


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code or dataset in your research, please cite the following doi:

