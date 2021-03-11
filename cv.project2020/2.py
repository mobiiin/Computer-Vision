import numpy as np
import cv2 as cv
matrix = np.array([[0,0,0,0,0]], dtype=object)
#matrix.append([])
#matrix.append([])

box = [3,10,5,16,120]

#matrix[1] = [box]
#matrix[1].append(box)


new = np.array([[1,2,3,4,50]], dtype=object)

matrix = np.vstack((matrix,new))

matrix[1].append([3,10,5,16,123])
print(matrix)
print(matrix.shape)

