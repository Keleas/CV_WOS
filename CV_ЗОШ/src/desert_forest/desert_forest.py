## Задачка для контеста школьникам.
## Требуется определить, что находится на изображении: пустыня или лес.

from PIL import Image
import numpy as np

## путь к картинке
path = os.getcwd()
folder = 'task_img'
path_img = input('Введите название изображения: ')

## загружаем изобржаение 
image = Image.open(os.path.join(path, folder, path_img))
image = np.array(image)

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