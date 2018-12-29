## Задачка для контеста школьникам.
## Требуется определить, что находится на изображении: пустыня или лес.

import numpy as np

## путь к картинке
path = 'C:\\Users\\user\\Python_Code\\contest_cv\\desert_forest'
folder = 'task_img\\arrays'

## загружаем изобржаение 
image = np.loadtxt('C:\\Users\\user\\Python_Code\\contest_cv\\desert_forest\\task_img\\arrays\\test_image_00.txt')
image = image.reshape((768, 768, 3))
image = image.astype(int)

## выделяем цветовые каналы
red = image[:, :, 0] 
green = image[:, :, 1] 
blue = image[:, :, 2] 

## процент цвета в канале
red_sq = red[red > 50].shape[0] / (768**2)
green_sq = green[green > 50].shape[0] / (768**2)

## проверка территории 
if green_sq > red_sq and green_sq > 0.5:
    print('Forest')
elif green_sq > 0.9 and red_sq > 0.9:
    print('Desert')