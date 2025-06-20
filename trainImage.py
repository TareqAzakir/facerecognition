import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image

# Train image data using LBPH algorithm (by Tareq Azakir)
def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message, text_to_speech):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcasecade_path)
    faces, Id = getImagesAndLabels(trainimage_path)
    recognizer.train(faces, np.array(Id))
    recognizer.save(trainimagelabel_path)
    res = "Training completed successfully."
    message.configure(text=res)
    text_to_speech(res)

# Helper function written while working on Tareq Azakir’s face recognition project
def getImagesAndLabels(path):
    newdir = [os.path.join(path, d) for d in os.listdir(path)]
    imagePaths = [
        os.path.join(newdir[i], f)
        for i in range(len(newdir))
        for f in os.listdir(newdir[i])
    ]
    faces = []
    Ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert("L")
        imageNp = np.array(pilImage, "uint8")
        try:
            Id = int(os.path.split(imagePath)[-1].split("_")[1])
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"[Tareq Log] Skipping file {imagePath}: {e}")

    return faces, Ids
