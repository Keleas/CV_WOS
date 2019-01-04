## Проверка решения участника.
## Задачка для контеста школьникам.
## Требуется посчитать площадь корабля на картинке в пикселях.

from PIL import Image
import numpy as np
import os

## решение участника
answer = int(input())


## путь к картинке 
path = os.getcwd()
folder = 'task_img'
path_img = input('Введите название изображения: ')

## загружаем изобржаение 
frame = Image.open(os.path.join(path, folder, path_img))
## переводим в grascale
frame_gray = frame.convert('L')
frame_gray = np.array(frame_gray)

## производим бинаизацию изображения
def threshold_(x, hold=100):
    return 255 if x > hold else 0
threshold = np.vectorize(threshold_)
## бинаризованная кртинка с корбаликом
thresh_max = threshold(frame_gray, hold=110)
thresh_min = threshold(frame_gray, hold=90)

## считаем площадь корабля в пикселях
square_max = (thresh_max/255).sum()
square_min = (thresh_min/255).sum()

if answer >= square_min and answer <= square_max:
	## верное решение 
    print('True')
else:
	## неверное решение
    print('False')
