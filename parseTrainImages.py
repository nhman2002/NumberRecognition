import numpy as np
import struct
import os
from PIL import Image

print(" In processing ...")

dataFile = r'.\MNIST_data\train-images-idx3-ubyte\train-images-idx3-ubyte'
dataFileSize = 47040016
dataFileSize = str(dataFileSize - 16) + 'B'
dataBuff = open(dataFile, 'rb').read()
magic, numImages, numRows, numCols = struct.unpack_from('>IIII', dataBuff, 0)
datas = struct.unpack_from('>' + dataFileSize, dataBuff, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numCols)

labelFile = r'.\MNIST_data\train-labels-idx1-ubyte\train-labels-idx1-ubyte'

labelFileSize = 60008
labelFileSize = str(labelFileSize - 8) + 'B'

labelBuff = open(labelFile, 'rb').read()

magic, numLabels = struct.unpack_from('>II', labelBuff, 0)

labels = struct.unpack_from('>' + labelFileSize, labelBuff, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

datas_root = 'mnist_train'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)
    
for id in range(10):
    fileName = datas_root + os.sep + str(i)
    if not os.path.exists(fileName):
        os.mkdir(fileName)
        
print("Processing Images ....")

for id in range(numLabels):
    if id % 100 == 0:
        print("Processed {id}/{numLabels} images ...")
    img = Image.fromarray(datas[id, 0, 0:28, 0:28])
    label = labels[id]
    fileName = datas_root + os.sep + str(label) + os.sep + 'mnist_train' + str(id) + '.png'
    img.save(fileName)
    
print("Processing Done!")