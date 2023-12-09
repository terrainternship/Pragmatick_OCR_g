import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd


class ImgWb:
    def __init__(self, path_img=None, threshold_wb=240):
        self.path_img = path_img  # путь к файлу с изображением
        self.threshold_wb = threshold_wb  # порог преобразования в бело-чёрное изображение

        self.img = None  # исходное изображение
        self.img_wb = None  # бело-чёрное изображение
        self.shape = None  # размер изображения
        self.center = None  # координаты центра изображения

        self.load_img()  # загружаем изображение
        self.get_wb_img()  # перевод в бело-чёрное изображение по порогу

    def load_img(self):
        # загружаем изображение
        if self.path_img is not None:
            self.img = cv2.imread(self.path_img)

    def set_img_wb(self, img_wb):
        self.img_wb = img_wb
        self.shape = img_wb.shape
        self.center = int(self.shape[0] / 2) + 1, int(self.shape[1] / 2) + 1

    def get_wb_img(self):
        # Перевод в бело-чёрное изображение по порогу
        if self.img is not None:
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # преобразуем в чёрно-белое
            img_wb = np.uint8((img < self.threshold_wb) * 255)  # преобразуем в бело-чёрное по порогу
            self.set_img_wb(img_wb)

    def contrast(self, threshold):
        # делаем из оттенков серого бело-чёрное изображение по порогу (добавляем контраста)
        self.img_wb = np.uint8((self.img_wb > threshold) * 255)

    def invert_img(self):
        """Инвертирует цвета чёрно-белое в бело-чёрное и наоборот"""
        self.img_wb = np.uint8((self.img_wb < 128) * 255)

    @staticmethod
    def create_black_img(height, width):
        # Создание чёрного изображения с размером shape
        im = ImgWb()
        arr = np.full((width, height), 0, np.uint8)
        im.set_img_wb(arr)  # создание чёрного фона
        return im

    @staticmethod
    def crop_img(img):
        # Обрезка пустых чёрных границ по периметру
        mask_w = np.any(img > 0, axis=0)  # маска по ширине
        ind_min_w = mask_w.argmax()  # минимальный индекс
        ind_max_w = len(mask_w) - mask_w[::-1].argmax()  # максимальный индекс

        mask_h = np.any(img > 0, axis=1)  # маска по ширине
        ind_min_h = mask_h.argmax()  # минимальный индекс
        ind_max_h = len(mask_h) - mask_h[::-1].argmax()  # максимальный индекс

        return img[ind_min_h:ind_max_h, ind_min_w:ind_max_w]

    def add_img(self, img, center):
        """
        Накладывает белое изображение с чёрным фоном (текущее) на другое белое изображение с чёрным фоном (im)
        :param img: вставляемое изображение
        :param center: координаты центра (y, x) для центра im
        :return: объединённое белое изображение с чёрным фоном
        """
        h1, w1 = self.img_wb.shape
        h2, w2 = img.shape

        shift_h = int(center[0] - (h2 + 1) / 2)
        shift_w = int(center[1] - (w2 + 1) / 2)

        # https://habr.com/ru/articles/528144/
        translation_matrix = np.float32([[1, 0, shift_w], [0, 1, shift_h]])  # создаём матрицу преобразований
        im_shift = cv2.warpAffine(img.img_wb, translation_matrix, (w1, h1))

        sum_img = np.maximum(self.img_wb, im_shift)  # наложение картинок
        new_img = ImgWb()
        new_img.set_img_wb(sum_img)
        return new_img

    def rotation_img(self, angle):
        """
        Поворот изображения на заданный угол
        :param angle: угол поворота
        :return: повёрнутое изображение
        """
        h, w = self.shape

        max_size = int(max(h, w) * 1.42)  # увеличиваем размер, чтобы влезло повёрнутое изображение

        fon = ImgWb().create_black_img(max_size, max_size)
        center = (int(max_size / 2), int(max_size / 2))
        im_center = fon.add_img(self, center)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        im_rotation = cv2.warpAffine(im_center.img_wb, rotation_matrix, (max_size, max_size))

        # Обрезка пустых чёрных границ по периметру
        im_rotation = self.crop_img(im_rotation)

        new_img = ImgWb()
        new_img.set_img_wb(im_rotation)
        return new_img


class DataSet:
    def __init__(self, path_dataset: str, path_marks: str, path_img_table: str, fon_size: tuple, params: dict,
                 num_imgs: int):
        """
        :param params:
        params = {
            'threshold_wb': 240,  # порог преобразования в бело-чёрное изображение
            'threshold_contrast': 100,  # порог контраста преобразования для конечного бело-чёрного изображения
             # Для конечного изображения
             'd_angle_max': 0.65,  # +- угол поворота
             'd_h_max': 0.1,  # +- максимальный коэффициент сдвига вверх/вниз относительно размера изображения по высоте
             'd_w_max': 0.012,  # +- максимальный коэффициент сдвига вправо/влево относительно размера изображения
                                # по ширине
             # Для метки
             'd_angle_mark_max': 20.0,  # +- угол поворота
             'zone_border_left_right': [-0.01,  0.3116,  0.6795,       1.01],   # относительные параметры границ зон
             #                            |  за  | против | воздержался |       # относительно размеров таблицы
             # параметры для текущей зоны относительно таблицы
             border_up = -0.3       # граница сверху таблицы
             border_down = 1.3      # граница снизу таблицы
        }
        """
        # Параметры
        self.path_dataset = path_dataset  # папка для сохранения датасета
        self.path_marks = path_marks  # папка с метками
        self.path_img_table = path_img_table  # путь к изображению таблицы
        self.fon_height, self.fon_width = fon_size  # высота и ширина фона
        self.params = params  # параметры генерации базы данных
        self.num_imgs = num_imgs  # общее кол-во изображений в датасете

        # Прочее
        self.img_fon_table_wb = None  # бело-чёрное изображение таблицы на фоне
        self.img_table_wb = None  # бело-чёрное изображение таблицы
        self.img_fon = None  # чёрный фон
        self.imgs_marks_wb = []  # бело-чёрные изображения меток

        self.zones = {'за': 0, 'против': 1, 'воздержался': 2}
        # self.categories = {'за': 'za', 'против': 'protiv', 'воздержался': 'vozderjalsa', 'испорчен': 'isporchen'}
        self.categories = {'за': 'za', 'против': 'protiv', 'воздержался': 'vozderjalsa',
                           'испорчен': 'isporchen', 'не голосовал': 'ne_golosoval'}

        self.create_imgs_marks_wb()  # Создание бело-чёрных изображений меток

        # Текущие переменные
        self.cur_category = None  # текущая обрабатываемая категория
        self.cur_number_imgs = None  # текущий номер таблицы для текущей категории
        self.cur_label = ['0', '0', '0']  # текущая метка (multilabel)
        self.cur_path_img = None  # путь к текущему изображению
        self.df_ds = None  # таблица соответствия сгенерируемого изображения и мультиметки

        self.create_dataset()  # Генерирует датасет в равных кол-вах для каждой категории

    def create_imgs_marks_wb(self):
        """Создание бело-чёрных изображений меток """
        for name_file in os.listdir(self.path_marks):
            path_mark = os.path.join(self.path_marks, name_file)
            self.imgs_marks_wb += [ImgWb(path_mark, self.params['threshold_wb'])]

    @staticmethod
    def imshow(name, im):
        cv2.imshow(name, im)
        cv2.waitKey(0)

    @staticmethod
    def clean_folder(path_folder):
        print(path_folder)
        if os.path.isdir(path_folder):
            shutil.rmtree(path_folder)
        os.mkdir(path_folder)

    def clean_dataset(self):
        """Очистка старого датасета, если он есть"""
        print('Удаление датасета')
        self.clean_folder(self.path_dataset)
        for _, category in self.categories.items():
            path = os.path.join(self.path_dataset, category)
            self.clean_folder(path)

    def create_dataset(self):
        """
        Генерирует датасет в равных кол-вах для каждой категории
        """
        # Очистка старого датасета, если он есть
        self.clean_dataset()
        self.img_fon = ImgWb.create_black_img(self.fon_height, self.fon_width)  # создание чёрного фона

        # Подготовка фона для генерации
        # бело-чёрное изображение таблицы
        self.img_table_wb = ImgWb(self.path_img_table, self.params['threshold_wb'])
        # self.imshow('img_table_wb', self.img_table_wb.img_wb)
        # cv2.imwrite(path_save_img + '/img_table_wb.png', img_table_wb)

        # Вставляем таблицу в фон
        self.img_fon_table_wb = self.img_fon.add_img(self.img_table_wb, self.img_fon.center)
        # imshow('img_fon_table_wb', self.img_fon_table_wb.img_wb)

        # Подготовка таблицы соответствия сгенерируемого изображения и мультиметки
        self.df_ds = pd.DataFrame(columns=['filepath', 'label_str'])

        # Цикл по категориям
        num_imgs_for_category = int(self.num_imgs / len(self.categories))
        for category_ru, category_en in self.categories.items():
            # Создаём изображения для выбранной категории
            print('----------------------------------------------')
            print(f'Генерация категории: {category_ru}')
            print('----------------------------------------------')
            self.cur_category = category_ru
            self.create_imgs_for_category(category_ru, num_imgs_for_category)

        # Сохраняем таблицу с мультиметками в файл
        path_file_csv = self.path_dataset + '/multilabel.csv'
        self.df_ds.to_csv(path_file_csv)

    def create_img_with_mark(self, zone):
        """
        Создаёт изображение со случайно выбранной меткой в случайном месте в пределах заданной зоны.
        Метка предварительно поворачивается на случайный угол, в пределах заданных в параметрах генерации.
        :param zone: зона для метки
        :return: бело-чёрное изображение метки на чёрном фоне
        """
        img_mark_wb = np.random.choice(self.imgs_marks_wb)  # случайный выбор метки из списка
        # Для метки
        d_angle_mark_max = self.params['d_angle_mark_max']  # +- угол поворота
        rnd = 2 * np.random.rand() - 1.0  # (-1.0, 1.0)
        d_angle_mark = rnd * d_angle_mark_max

        # Вращаем метку
        img_mark_wb_rotation = img_mark_wb.rotation_img(d_angle_mark)  # при повороте изменяется размер изображения
        # self.imshow('img_fon_mark', img_mark_wb_rotation.img_wb)
        h_mark, w_mark = img_mark_wb_rotation.shape  # размеры повёрнутой метки

        # параметры для текущей зоны относительно таблицы
        border_left = self.params['zone_border_left_right'][self.zones[zone]]
        border_right = self.params['zone_border_left_right'][self.zones[zone] + 1]
        border_up = self.params['border_up']
        border_down = self.params['border_down']

        fon_height = self.img_fon.shape[0]
        fon_width = self.img_fon.shape[1]
        table_height = self.img_table_wb.shape[0]
        table_width = self.img_table_wb.shape[1]
        # верхняя граница центра метки относительно фона в px
        border_center_up_fon_px = 0.5 * (fon_height + table_height * (2 * border_up - 1)) + h_mark / 2
        # нижняя граница центра метки относительно фона в px
        border_center_down_fon_px = 0.5 * (fon_height + table_height * (2 * border_down - 1)) - h_mark / 2
        # левая граница центра метки относительно фона в px
        border_center_left_fon_px = 0.5 * (fon_width + table_width * (2 * border_left - 1)) + w_mark / 2
        # правая граница центра метки относительно фона в px
        border_center_right_fon_px = 0.5 * (fon_width + table_width * (2 * border_right - 1)) - w_mark / 2

        # генерируем случайную позицию в фоне для центра метки в пределах рассчитанных границ
        rnd = np.random.rand()  # (0.0, 1.0)
        d_border_center_fon_px = border_center_down_fon_px - border_center_up_fon_px
        if d_border_center_fon_px > 0.0:
            position_center_h = border_center_up_fon_px + rnd * d_border_center_fon_px
        else:
            position_center_h = fon_height / 2

        rnd = np.random.rand()  # (0.0, 1.0)
        d_border_center_fon_px = border_center_right_fon_px - border_center_left_fon_px
        if d_border_center_fon_px > 0.0:
            position_center_w = border_center_left_fon_px + rnd * d_border_center_fon_px
        else:
            position_center_w = fon_width / 2

        position_center_mark = int(position_center_h), int(position_center_w)

        # Получаем метку на фоне
        img_fon_mark = self.img_fon.add_img(img_mark_wb_rotation, position_center_mark)
        # self.imshow('img_fon_mark', img_fon_mark.img_wb)

        return img_fon_mark

    def create_imgs_for_zones(self, zones, num_imgs):
        """
        Создаёт изображения для выбранных зон
        :param zones: список зон для меток
        :param num_imgs: кол-во изображений для выбранных зон
        """
        print(f'зоны: {zones}')
        self.cur_label = ['0', '0', '0']  # текущая метка
        # цикл по текущим зонам
        for z in zones:
            index_zone_for_label = self.zones[z]  # индекс зоны для метки
            self.cur_label[index_zone_for_label] = '1'

        # Цикл по кол-ву изображений
        for _ in tqdm(range(num_imgs)):
            self.cur_number_imgs += 1  # текущий номер таблицы для текущей категории (нумерация файлов в категории с 1)
            # Цикл по зонам
            img_table_marks = self.img_fon_table_wb  # изображение меток на таблице
            for zone in zones:
                img_fon_mark = self.create_img_with_mark(zone)  # изображение метки на чёрном фоне
                img_table_marks = img_table_marks.add_img(img_fon_mark, img_table_marks.center)

            # Поворачиваем и смещаем (аугментируем) конечное изображение
            rnd = 2 * np.random.rand() - 1.0  # (-1.0, 1.0)
            d_angle = rnd * self.params['d_angle_max']
            rnd = 2 * np.random.rand() - 1.0  # (-1.0, 1.0)
            d_h = rnd * self.params['d_h_max']
            rnd = 2 * np.random.rand() - 1.0  # (-1.0, 1.0)
            d_w = rnd * self.params['d_w_max']

            img_all_rotation = img_table_marks.rotation_img(d_angle)  # при повороте изменяется размер изображения
            # сдвигаем вверх/вниз вправо/влево относительно размера изображения
            h, w = self.img_fon.shape
            dh_px = int(d_h * h)
            dw_px = int(d_w * w)
            position_center_table_marks = self.img_fon.center[0] + dh_px, self.img_fon.center[1] + dw_px
            img_end = self.img_fon.add_img(img_all_rotation, position_center_table_marks)
            # преобразуем из оттенков серого в бело-чёрное изображение по порогу (добавляем контраста)
            img_end.contrast(self.params['threshold_contrast'])
            img_end.invert_img()  # инвертируем цвет

            # Сохраняем изображение
            # self.cur_path_img = os.path.join(self.path_dataset, self.categories[self.cur_category],
            #                                  str(self.cur_number_imgs) + '.png')
            self.cur_path_img = \
                f'{self.path_dataset}/{self.categories[self.cur_category]}/{str(self.cur_number_imgs)}.png'

            # Формируем таблицу с метками
            # print(f'{self.cur_path_img} : {self.cur_label}')
            label = ''.join(self.cur_label)
            new_row = pd.DataFrame([{'filepath': self.cur_path_img, 'label_str': label}])
            self.df_ds = pd.concat([self.df_ds, new_row], ignore_index=True)

            # Сохраняем конечное изображение
            cv2.imwrite(self.cur_path_img, img_end.img_wb)

    def create_imgs_for_category(self, category, num_imgs):
        """
        Создаёт изображения для выбранной категории
        :param category: категория
        :param num_imgs: кол-во изображений в категории
        """
        # Перебор по зонам
        num = num_imgs
        list_zones = [[]]
        if category in ['за', 'против', 'воздержался']:
            list_zones = [[category]]
        elif category in ['не голосовал']:
            pass
        elif category in ['испорчен']:
            list_zones = [['за', 'против'],
                          ['за', 'воздержался'],
                          ['против', 'воздержался'],
                          ['за', 'против', 'воздержался']]
            num = int(num_imgs / len(list_zones))  # кол-во изображений для каждого списка зон
        self.cur_number_imgs = 0
        for zs in list_zones:
            self.create_imgs_for_zones(zs, num)

    @staticmethod
    def RGB_to_bw(img, max_percentage_of_filling=12.0, step_threshold=2):
        """Конвертация RGB-изображения в чёрно-белое
        :param img: входное изображение
        :param max_percentage_of_filling: (в %) максимальный процент заполнения искомым изображением на фоне
        :param step_threshold: шаг поиска границы threshold_wb
        """
        threshold_wb = 5
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # преобразуем в чёрно-белое
        # Динамический подбор параметров threshold_wb
        # Определяется как среднее значение numpy-массива изображения
        threshold_wb = np.average(img_gray)  # в первом приближении
        img_wb = None
        percentage_of_filling = 100.0  # в %. Стартовое значение для цикла
        while percentage_of_filling > max_percentage_of_filling:  # достигли, требуемого значения
            # img_wb = np.uint8((img_gray < threshold_wb) * 255)  # преобразуем в бело-чёрное по порогу
            img_wb = np.uint8((img_gray > threshold_wb) * 255)  # преобразуем в чёрно-белое по порогу
            # DataSet.imshow(str(percentage_of_filling), img_wb)
            # Считаем процент чёрного в чёрно-белом изображении. Для хорошего изображения без лишнего шума должен быть
            # меньше заданного max_percentage_of_filling
            percentage_of_filling = np.sum(img_wb == 0) * 100.0 / img_wb.size  # в %
            threshold_wb -= step_threshold
            if threshold_wb < 0:  # уменьшать дальше нельзя, выходим из цикла
                break
        return img_wb
