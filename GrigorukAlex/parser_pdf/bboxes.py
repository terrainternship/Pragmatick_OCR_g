"""
    Статья: https://habr.com/ru/companies/ruvds/articles/765246/

    https://pypi.org/project/pdf2image/

    Для Windows:
        Для работы требуется установленная программа poppler и добавлен путь до папки 'bin' в переменную окрежения
        PATH="C:\path\to\poppler-xx\bin".
        Скачать poppler архив для Windows можно здесь https://github.com/oschwartz10612/poppler-windows/releases/.
        Так же лежит в папке distrib этого модуля.
        Если ругается на poppler, следует проверить запуск C:\path\to\poppler-xx\bin\pdfinfo из командной строки.
        Если ругается на отсутствие файла .dll, то установить VC_redist.x64 и VC_redist.x86 из distrib.

    Для Linux:
        sudo apt update
        sudo apt install xpdf
        pip install pdfplumber
        pip install pdf2image

"""
# Для извлечения текста из таблиц в PDF
import pdfplumber
# Для работы с изображениями
import cv2
# Для извлечения изображений из PDF
from pdf2image import convert_from_path

import os
import shutil
import json


# Очищаем папку со всем содержимым
def clean_dir_tree(folder):
    # Удаляем папку с одноимённым изображением, если она уже есть
    try:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (folder, e))
    # Создаём папку с одноимённым изображением
    os.mkdir(folder)


# Преобразование PDF в изображения и сохранение в одноимённую папку
def convert_to_images(pdf_path, dpi):
    # Преобразовываем PDF в изображения
    imgs = convert_from_path(pdf_path, dpi=100)
    # Определяем путь к одноимённой папке
    folder, _ = os.path.splitext(pdf_path)
    # Очищаем папку со старыми изображениями, если она есть
    clean_dir_tree(folder)
    for page_num, img in enumerate(imgs):
        name_img = 'page_' + str(page_num + 1) + '.png'
        output_file = os.path.join(folder, name_img)
        img.save(output_file, "PNG")

    return folder


# Отображение изображения
def cv2_show(name, img):
    cv2.imshow(name, img)  # Для компа
    # cv2_imshow(img)  # Для колаба
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Извлечение bounding_boxes таблиц из страницы
def get_bboxes_tables_page(pdf_path, skip_num_tables, padding):
    """
    :param pdf_path:
    :param skip_num_tables: кол-во таблиц на первой странице, которые не будут включены в найденные bboxes
    :param padding: Отступ слева, сверху, справа, снизу от границы таблицы в % от ширины и высоты страницы
        Пример: padding=(10.0, 5.0, 4.0, 6.0)
    :return:
    """
    # Определяем размеры страницы
    size_page = get_size_page(pdf_path)
    print(size_page)

    # Открываем файл pdf
    pdf = pdfplumber.open(pdf_path)
    # Определяем кол-во страниц
    num_pages = len(pdf.pages)
    print('Кол-во страниц в PDF: ', num_pages, ' | Файл: ', pdf_path)

    # Берём первую страницу
    bbox = pdf.pages[0].bbox
    # Определяем ширину и высоту страницы в PDF размерах
    width, height = int(bbox[2]), int(bbox[3])

    # Определяем отступы
    padding_left = padding[0] / 100.0  # отступ слева в долях по ширине
    padding_top = padding[1] / 100.0  # отступ сверху в долях по высоте
    padding_right = padding[2] / 100.0  # отступ справа в долях по ширине
    padding_bottom = padding[3] / 100.0  # отступ снизу в долях по высоте

    num_pages = len(pdf.pages)

    # Словарь с bounding_boxes для всех страниц
    bboxes_pages = {'num_pages': num_pages,  # кол-во страниц
                    'pages': {}  # страницы
                    }
    # Цикл по страницам
    for page_num, page in enumerate(pdf.pages):
        # Извлекаем соответствующую таблицу
        # tables = page.extract_tables()
        # Находим координаты таблиц на текущей странице
        tables = page.find_tables()

        # Удаляем первые 4 таблице у первого листа
        if page_num == 0 and len(tables) > 3:
            tables = tables[skip_num_tables:]
        print('Кол-во таблиц на странице ', page_num + 1, ' : ', len(tables), ' шт')

        # Словарь с bounding_boxes для одной страницы
        bboxes_page = {'num_tables': len(tables),  # кол-во таблиц на странице
                       'tables': {}  # таблицы
                       }
        # Извлекаем координаты таблиц для текущей страницы
        bboxes_tables = {}
        for table_num, table in enumerate(tables):
            bbox = table.bbox
            left = bbox[0] / width - padding_left
            top = bbox[1] / height - padding_top
            right = bbox[2] / width + padding_right
            bottom = bbox[3] / height + padding_bottom
            if left < 0.0:
                left = 0.0
            if top < 0.0:
                top = 0.0
            if right > 1.0:
                right = 1.0
            if bottom > 1.0:
                bottom = 1.0
            bbox_norm = {'left': left,
                         'top': top,
                         'right': right,
                         'bottom': bottom}
            bboxes_page['tables'][table_num + 1] = bbox_norm
        bboxes_pages['pages'][page_num + 1] = bboxes_page

    print(bboxes_pages)
    return bboxes_pages


# Получить размеры страницы
def get_size_page(pdf_path):
    # Открываем файл pdf
    pdf = pdfplumber.open(pdf_path)
    # Находим исследуемую страницу
    table_page = pdf.pages[0]
    bbox = table_page.bbox
    width, height = int(bbox[2]), int(bbox[3])
    # Возвращаем размеры страницы
    return width, height


# Нарисовать прямоугольник на изображении
def draw_rect_img(img, rect):
    """
    :rect: [x1, y1, x2, y2] координаты левого верхнего и правого нижнего углов
    """
    cv2.rectangle(img,
                  (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])),
                  (0, 255, 0),
                  2)  # Рисуем найденный контур


# Отрисовываем все bboxes на всех изображениях
def draw_bboxes(path_imgs, bboxes_pages):
    """
    :param path_imgs: путь к папке с изображениями
    :param bboxes_pages: словарь bboxes для всех изображений в относительных значениях (от 0 до 1)
    :return:
    """
    num_pages = bboxes_pages['num_pages']  # кол-во страниц
    # Цикл по стриницам
    for page_num in range(num_pages):
        # Открываем изображение текущей страницы PDF
        page_num_str = str(page_num + 1)
        name_img_page = 'page_' + page_num_str + '.png'  # имя файла изображения PDF
        path_img_page = os.path.join(path_imgs, name_img_page)  # путь к файлу изображения PDF
        img_page = cv2.imread(path_img_page)
        # Определяем размеры изображения
        width, height = img_page.shape[1], img_page.shape[0]

        # Отрисовываем bbox на изображениях
        bboxes_page = bboxes_pages['pages'][page_num_str]  # словарь bboxes для текущей страницы
        num_tables = bboxes_page['num_tables']  # кол-во таблиц
        bboxes_tables = bboxes_page['tables']  # словарь с таблицами для текущей страницы
        # Цикл по таблицам
        for num_table in range(num_tables):
            num_table_str = str(num_table + 1)
            bbox_table = bboxes_tables[num_table_str]  # словарь bbox для текущей таблицы

            left = bbox_table['left'] * width
            top = bbox_table['top'] * height
            right = bbox_table['right'] * width
            bottom = bbox_table['bottom'] * height
            bbox = (left, top, right, bottom)

            draw_rect_img(img_page, bbox)

        # Сохраняем изображение
        cv2.imwrite(path_img_page, img_page)
        # cv2_show("Image", img_page)


def parse(pdf_path, skip_num_tables, padding, dpi=100, flag_draw_bboxes=True):
    """
    Извлекает из PDF-файла (pdf_path) координаты bboxes для всех страниц файла. Сохраняет координаты в JSON-файл
    с таким же именем как у PDF-файл в этой же папке.
    Все страницы PDF-документа конвертятся в PNG-формат и сохраняются тут же в новой папке с тем же название как
    у PDF-файла. Для наглядного представления все bboxes отрисовываются на полученных изображениях.
    Координаты сохраняются в размерах от 0 до 1 относительно высоты и ширины документа.
    :param pdf_path: путь к файлу PDF
    :param skip_num_tables: кол-во таблиц на первой странице, которые не будут включены в найденные bboxes
    :param padding: Отступ сверху, снизу и слева, справа от границы таблицы в % от ширины и высоты страницы
    :param dpi: кол-во точек на дюйм в полученном изображении при конвертации PDF в PNG
    :param flag_draw_bboxes: True - для отрисовки bboxes на изображениях, False - без отрисовки
    Пример: padding=(10.0, 5.0, 4.0, 3.0)
    :return:
    """
    path_imgs = convert_to_images(pdf_path, dpi)

    # Создаём пустое белое изображение по размеру страницы
    # image = np.full((height, width, 3), 255, np.uint8)

    # Извлекаем bounding_boxes таблиц из страниц
    bboxes_pages = get_bboxes_tables_page(pdf_path, skip_num_tables, padding)

    # Сохраняем bboxes в json-файл
    path_settings_table, _ = os.path.splitext(pdf_path)
    path_settings_table_json = path_settings_table + '.json'
    with open(path_settings_table_json, 'w') as file:
        json_string = json.dumps(bboxes_pages, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        file.write(json_string)

    # Загружаем bboxes из файла
    with open(path_settings_table_json, 'r') as json_file:
        bboxes_pages = json.load(json_file)

    # Если установлен флаг отрисовки
    if flag_draw_bboxes:
        # Отрисовываем все bboxes на всех изображениях
        draw_bboxes(path_imgs, bboxes_pages)
