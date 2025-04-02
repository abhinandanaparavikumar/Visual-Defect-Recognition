'''
<This program is a visual defect identification program>
    Copyright (C) <2019>  <Abhinanda Napa Ravikumar>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import glob
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.models import model_from_json
import keras
from keras import utils as np_utils




class members:
        def __init__(self, name, IF, imgforele = [], defectele=[], DR = 0):
                self.name = name
                self.IF = IF
                self.imgforele = imgforele
                self.defectele = defectele
                self.DR = DR
        def load_images_from_folder(folder):
                images = []
                for filename in os.listdir(folder):
                        img = cv2.imread(os.path.join(folder,filename))
                        if img is not None:
                                img = cv2.resize(img, (180,180))
                                images.append(img)
                return images
        def defect_identification():
                m=0
                img_array = []
                predictions = []
                indexes = []
                indexes1 = []
                for k in range(0, nofdiffele):
                        for l in range (0, len(mem[k].imgforele)):
                                img_array = tf.keras.utils.img_to_array(mem[k].imgforele[l])
                                img_array = tf.expand_dims(img_array, 0)
                                predictions = model.predict(img_array)
                                indexes = np.argmax(predictions)
                                print (" For "+ str(mem[k].name)+" Img no. ", l+1)
                                if indexes == 0:
                                        print(" 0 - CORROSION")
                                elif indexes == 1:
                                        print(" 1 - CRACK")
                                indexes1.append(indexes)
                                m+=1
                return indexes1
        def defect_input(defect, nofdiffele, mem):
                p = 0
                for k in range(0, nofdiffele):
                        for l in range(0, len(mem[k].imgforele)):
                                if defect[p] == 0:
                                        mem[k].defectele.append("CORROSION")
                                        p+=1
                                        break
                                elif defect[p] == 1:
                                        mem[k].defectele.append("CRACK")
                                        p+=1
                                        break

                for m in range (0, nofdiffele):
                        if len(mem[m].defectele) == 1:
                                mem[m].DR = 0.2
                        elif len(mem[m].defectele) == 2:
                                mem[m].DR = 0.4
                        elif len(mem[m].defectele) == 3:
                                mem[m].DR = 0.6
                        elif len(mem[m].defectele) == 4:
                                mem[m].DR = 0.8
                        elif len(mem[m].defectele) >= 5:
                                mem[m].DR = 1
                        
                
                

                
nofdiffele = int(input("enter number of diff ele: "))

mem = []
for i in range(0, nofdiffele):
        pathtemp = str(input("Enter the path folder with / to imgs of the element: "))
        mem.append( members( str(input("Name of element: ")), float(input("importance factor: ")), imgforele = members.load_images_from_folder(pathtemp)))

print(os.getcwd())

model = tf.keras.models.load_model("img_class.model")

defect = members.defect_identification()

members.defect_input(defect, nofdiffele, mem)

for j in range(0, nofdiffele):
        print("name: " + str(mem[j].name))
        print("Importance Factor: ", mem[j].IF)
        for k in range (0, len(mem[j].imgforele)):
                cv2.imshow(str(mem[j].defectele[k]), mem[j].imgforele[k])
        print("Defects are: ")
        print(mem[j].defectele)
        print("Defect Rating: ", mem[j].DR)







