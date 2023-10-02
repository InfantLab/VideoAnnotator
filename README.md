# Baby Jokes Video Analysis
## Caspar Addyman <infantologist@gmail.com>

A demonstration project using machine learning models to analyse dataset of videos of parents demonstrating jokes to babies. This dataset was assembled for Sage Ethical AI hackathon 2023. It serves as a small test case to explore challenges with machine learning models of parent child interactions. You can watch a video motivating the project here [Sage Hackathon 2023 - PCI Video Analysis 6m20](https://www.youtube.com/watch?v=mt0Um-ZNbj4)

## Dataset
A small test dataset is provided in the `LookitLaughter.test` folder. It consists of 54 videos of parents demonstarting simple jokes to their babies. Metadata is provided in `_LookitLaughter.xlsx`. Each video shows one joke from a set of five possibilities [Peekaboo,TearingPaper,NomNomNom,ThatsNotAHat,ThatsNotACat]. For each joke parents rated how funny the child found it  [Not Funny, Slightly Funny, Funny, Extremely Funny] and whether they laughed [Yes, No]
*A larger dataset with 1425 videos is available on request.* 


## Code
All notebooks and supporting code are in the `code` folder. The numbered notebooks should be run in order to process the data, train the models and generate the results.

#TODO - visualise data
#TODO - build models & analysis


## Installation / Key Requirements

This project makes use of the following libraries and versions:

Python 3.11
Pytorch 2.0.1
ultralytics 8.0  (wrapper for YOLOv8 object detection model)
deepface 0.0.68 (Facial Expression Recognition)
speechbrain 0.5  (Speech Recognition)
openai-whisper (Opensource version of OpenAI's Whisper model)


Conda environment file is provided. To create the environment and activate it run the following commands in the terminal:
```
conda env create --name babyjokes --file=environment.yml
conda activate babyjokes
```
Using Pip
```
pip install -r requirements.txt
```

## Sage Hackathon
Sage data scientist, Yu-Cheng has a write up of his team's approach to the problem on the Sage-AI blog. [Quantifying Parent-Child Interactions: Advancing Video Understanding with Multi-Modal LLMs](https://medium.com/sage-ai/unlocking-parent-child-interactions-advancing-video-understanding-with-multi-modal-llms-c570ab487183)
Repositories from the hackathon are found here:
 * London team - Combining Speech recognition and laughter detection https://github.com/chilledgeek/ethical_ai_hackathon_2023
 * US team - Interpreting Parent laughter with VideoLLama https://github.com/yutsai84/Ask-Anything 
