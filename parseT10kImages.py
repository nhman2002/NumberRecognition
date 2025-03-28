import numpy as np
import struct
import os
from PIL import Image

print("In Processing...")

dataFile = r'.\MNIST_data\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte'

dataFileSize = 7840016
dataFileSize = str(dataFileSize - 16) + 'B'
dataBuff = open(dataFile, 'rb').read()
magic, numImages, numRows, numCols = struct.unpack_from('>IIII', dataBuff, 0)
datas = struct.unpack_from('>' + dataFileSize, dataBuff, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numCols)

labelFile = r'.\MNIST_data\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte'
labelFileSize = 10008
labelFileSize = str(labelFileSize - 8) + 'B'
labelBuff = open(labelFile, 'rb').read()
magic, numLabels = struct.unpack_from('>II', labelBuff, 0)
labels = struct.unpack_from('>' + labelFileSize, labelBuff, struct.calcsize('>II'))
labels = np.array(labels).astype(np.uint8)

datasRoot = 'mnist_test'
if not os.path.exists(datasRoot):
    os.mkdir(datasRoot)
    
for id in range(10):
    fileName = datasRoot + os.sep + str(id)
    if not os.path.exists(fileName):
        os.mkdir(fileName)
        
print("Processing Images...")

for i in range(numLabels):
    if i % 100 == 0:
        print(f"Processed {i}/{numLabels} images...")
    img = Image.fromarray(datas[i, 0, 0:28, 0:28])
    label = labels[i]
    fileName = datasRoot + os.sep + str(label) + os.sep + 'mnist_test' + str(i) + '.png'
    img.save(fileName)
    
print("Processing Done!")

