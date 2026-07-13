import numpy as np
from numpy.matrixlib.defmatrix import matrix
import matplotlib.pyplot as plt


loaded = np.load("endeavour.npz")
task1 = loaded["task1"]
matrix=task1



def rotate_image(matrix, degree):
    match degree:
        case 90:
            matrix = np.rot90(matrix, k = -1)
        case 180:
            matrix = np.rot90(matrix, k = -2)
        case 270:
            matrix = np.rot90(matrix, k = -3)
    return matrix


def plot_first_pattern(matrix):
    stf_arr = matrix[0]
    plt.imshow(stf_arr, cmap = 'Greys', interpolation = 'nearest')
    plt.show()


def count_of_rotations(matrix):
    ROTATIONS=[0,90,180,270]
    res=0
    result=[]
    for i in ROTATIONS:
        goal = rotate_image(matrix[0], i)
        for image in matrix:
            if np.array_equal(image,goal):
                res+=1
        result.append(res)
        res = 0
    for i in range(4):
        print(f"count of {ROTATIONS[i]} : {result[i]}")

#FIRST
DEGREE=90
for i in range(len(matrix)):
    matrix[i] = rotate_image(matrix[i],DEGREE)


#SECOND
count_of_rotations(matrix)



plot_first_pattern(matrix)





