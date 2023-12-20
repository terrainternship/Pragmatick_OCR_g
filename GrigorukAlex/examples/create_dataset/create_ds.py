from GrigorukAlex.create_dataset.dataset import DataSet

num_imgs = 200000
path_dataset = 'ds/create_ds/ds_multilabel_' + str(num_imgs)  # папка для сохранения датасета
path_marks = 'ds/create_ds/marks'  # папка с метками
path_img_table = 'ds/create_ds/img_table.png'  # путь к изображению таблицы
fon_size = (690, 35)  # высота и ширина фона
params = {
    'threshold_wb': 240,  # порог преобразования в бело-чёрное изображение
    'threshold_contrast': 100,  # порог контраста преобразования для конечного бело-чёрного изображения
    # Для конечного изображения
    'd_angle_max': 0.65,  # +- угол поворота
    'd_h_max': 0.1,  # +- максимальный коэффициент сдвига вверх/вниз относительно размера изображения по высоте
    'd_w_max': 0.012,  # +- максимальный коэффициент сдвига вправо/влево относительно размера изображения по ширине
    # Для метки
    'd_angle_mark_max': 20.0,  # +- угол поворота
    'zone_border_left_right': [-0.01,  0.3116,  0.6795,       1.01],  # относительные параметры границ зон
    #                            |  за  | против | воздержался |      # относительно размеров таблицы
    # параметры для текущей зоны относительно таблицы
    'border_up': -0.1,  # граница сверху таблицы
    'border_down': 1.1  # граница снизы таблицы
}

ds = DataSet(path_dataset, path_marks, path_img_table, fon_size, params, num_imgs)
