# Baby Jokes Video Analysis
## Caspar Addyman <infantologist@gmail.com>

A demonstration project using machine learning models to analyse dataset of videos of parents demonstrating jokes to babies. 


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