import json
import cv2

from GrigorukAlex.parser_pdf.bboxes import cut_bbox_from_img

# Путь к изображению из которого вырезаем bbox
path_img = 'db/page_1.png'

# Получаем к JSON-файлу с bboxes
path_bboxes = 'db/testPDF.json'

# Загружаем bboxes из файла
with open(path_bboxes, 'r') as j:
    bboxes_pages = json.load(j)

# Получаем нужный bbox
bbox = bboxes_pages['pages']['1']['tables']['1']

# Получаем вырезанное изображение из bbox
img_bbox = cut_bbox_from_img(path_img, bbox)

# Сохраняем изображение
cv2.imwrite('result/cut_bbox.png', img_bbox)
