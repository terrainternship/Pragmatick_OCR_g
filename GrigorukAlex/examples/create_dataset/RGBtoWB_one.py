from GrigorukAlex.create_dataset.dataset import DataSet
import cv2

path_img = 'ds/RGBtoBW_one/104.JPG'  # тестовое изображение
# path_img = 'ds/RGBtoBW_one/110.JPG'
img = cv2.imread(path_img)
DataSet.imshow('', img)     # просмотр тестового изображение

# Конвертация RGB-изображения в чёрно-белое
img_wb = DataSet.RGB_to_bw(img, max_percentage_of_filling=12.0, step_threshold=2)
DataSet.imshow('', img_wb)  # просмотр результата конвертации изображение
