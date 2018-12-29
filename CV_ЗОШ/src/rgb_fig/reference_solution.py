## Задачка для контеста школьникам.
## Требуется посчитать сторону квдрата и правильного треугольника,
## радиус окружности на изображении. 
## Фигуры всегда одного цвета.

import numpy as np
import math

## путь к картинке 
path = 'C:\\Users\\user\\Python_Code\\contest_cv\\rgb_fig'
folder = 'task_img\\arrays'

## загружаем изобржаение 
image = np.loadtxt('C:\\Users\\user\\Python_Code\\contest_cv\\rgb_fig\\task_img\\arrays\\test_image_00.txt')
image = image.reshape((1024, 1024, 3))
image = image.astype(int)


## вычленяем фигуры в цветовых каналах
red = image[:, :, 0] ## квадрат 
green = image[:, :, 1] ## круг
blue = image[:, :, 2] ## треугольник 

## площадь каждой фигуры 
red_sq = red[red > 0].shape[0]
blue_sq = blue[blue > 0].shape[0]
green_sq = green[green > 0].shape[0]


## радиус круга
r = (green_sq/math.pi) ** 0.5
## сторона квадрата 
a = red_sq ** 0.5
## сторона правильного треуглоьника 
b = (2*blue_sq)**0.5

print("""{:.2f}\n{:.2f}\n{:.2f}""".format(r, a, b))