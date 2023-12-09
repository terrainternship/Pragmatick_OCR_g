import os
# import numpy as np

from GrigorukAlex.create_dataset.dataset import DataSet
import cv2
from tqdm import tqdm

paths = ['za', 'protiv', 'vozderjalsa', 'isporchen', 'ne_golosoval']
for path in paths:
    print('Конвертация:', path)
    path_imgs = 'ds/RGBtoBW_all/db/' + path  # папка с изображениями для конвертации
    path_imgs_bw = 'ds/RGBtoBW_all/db/' + path + '_bw'  # папка для сконвертированных изображений
    list_path_imgs = os.listdir(path_imgs)      # список файлов входных изображений
    DataSet.clean_folder(path_imgs_bw)  # очистка директории

    # Цикл по входным изображениям
    for file_img in tqdm(list_path_imgs):
        path_img = os.path.join(path_imgs, file_img)
        img = cv2.imread(path_img)  # получаем одно входное изображение

        img_resize = cv2.resize(img, (690, 35))     # изменяем размер до требуемого
        # Конвертация RGB-изображения в чёрно-белое с заданным порогом
        img_wb = DataSet.RGB_to_bw(img_resize, max_percentage_of_filling=12.0, step_threshold=2)
        # алгоритм Оцу
        # (thresh, img_wb_OTSU) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # DataSet.imshow('', img_wb)
        img_wb_RGB = cv2.cvtColor(img_wb, cv2.COLOR_GRAY2RGB)
        # img_two = np.concatenate([img_resize, img_wb_RGB], axis=0)  # Для отображения на одной картинке до и после

        # Сохраняем изображение
        path_img_wb = os.path.join(path_imgs_bw, file_img)
        cv2.imwrite(path_img_wb, img_wb_RGB)
