import cv2
import os
from pathlib import Path
from tqdm import tqdm


fakeImages="images/fakeTestImages/"
realImages="images/realTestImages/"

fakeProcessedImages="processedImages/processedImagesValidation/validationFake"
realProcessedImages="processedImages/processedImagesValidation/validationReal"

os.makedirs(fakeProcessedImages, exist_ok=True)
os.makedirs(realProcessedImages, exist_ok=True) 

imageSize=224


def processFolder(inputFolder, outputFolder):
    for imageName in tqdm(os.listdir(inputFolder)):
        imagePath = os.path.join(inputFolder, imageName)
        img = cv2.imread(imagePath)
        if img is not None:
            img = cv2.resize(img, (imageSize, imageSize))
            outputPath = os.path.join(outputFolder, Path(imageName).stem + ".jpg") 
            cv2.imwrite(outputPath, img)

print("Processing fake images:")

processFolder(fakeImages, fakeProcessedImages)

print("Processing real images:")

processFolder(realImages, realProcessedImages)        