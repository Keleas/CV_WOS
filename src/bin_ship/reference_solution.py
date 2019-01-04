## Задачка для контеста школьникам.
## Требуется посчитать площадь корабля на картинке в пикселях.

import numpy as np

path = 'C:\\Users\\user\\Python_Code\\contest_cv\\bin_ship'
folder = 'task_img\\arrays'

## загружаем изобржаение
frame_gray = np.loadtxt('C:\\Users\\user\\Python_Code\\contest_cv\\bin_ship\\task_img\\arrays\\test_image_00.txt')
frame_gray = frame_gray.reshape((768, 768, 3))
frame_gray = frame_gray.astype(int)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

frame_gray = rgb2gray(frame_gray)

## бинаизация изображения по пороговому значению hold
def threshold_(x, hold=100):
    return 255 if x > hold else 0
threshold = np.vectorize(threshold_)

## бинаризованная кртинка с корбаликом
thresh = threshold(frame_gray)

## считаем площадь корабля в пикселях
square = (thresh/255).sum()

print("{}".format(square))