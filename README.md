# Crown of Thorns Starfish Object Detection

This repository contains the code to train an object-detection ML Model for detecting Crown-of-Thorns Starfish, an invasive species on the Great Barrier Reef

## The Model

The model used RetinaNet, a commonly used Deep Learning Object Detection algorithm which has been fine-tuned for this use case. It is written in Tensorflow.

## Repository Structure

```
crown-of-thorns
├─ .gitignore
├─ Dockerfile
├─ README.md
├─ dockercommands.txt
├─ main.py            - main script to run prepreprocessing and training
├─ requirements.txt
└─ src
   ├─ deployment.py 
   ├─ preprocessing.py - preprocessing functionality
   ├─ train.py         - training script
   ├─ utils
   │  ├─ config.py -GCP config
   │  ├─ model.py - contains RetinaNet Model
   │  └─ utils.py - contains 
   └─ viz.ipynb   - notebook for visualising results

```