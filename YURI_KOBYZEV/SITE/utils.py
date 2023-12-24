import os
import cv2
import shutil
import pathlib
import numpy as np
import cv2.typing as cvtp
import classes.common as cmn

def show_cv2_image(title: str, image: cvtp.MatLike):
    """Процедура отображения изображения с помощью cv2

    Args:
        title (str, optional): Название изображения.
        img (numpy.ndarray): Данные изображения.
    """
    
    cv2.imshow(title, image) # Для компа
    # cv2_imshow(image)  # Для колаба
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_crosses(image_src: cvtp.MatLike,
                 points:[cmn.Point],
                 size: int = 5,
                 thickness: int = 1,
                 color: cvtp.Scalar = (0, 255, 0),
                 relative: bool = True
                 )->cvtp.MatLike:
    """Отрисовка перекрестий в заданных точках

    Args:
        image_src (cvtp.MatLike): Исходное изображение
        points ([Point]): Массив заданных точек
        size (int, optional): Длина луча перекрестия в пикселях. Defaults to 5.
        thickness (int, optional): Толщина линий перекрестий. Defaults to 1.
        color (cvtp.Scalar, optional): Цвет перекрестий. Defaults to (0, 255, 0).
        relative (bool, optional): Флаг относительности координат:
            True - относительные, False - абсолютные. Defaults to True.

    Returns:
        cvtp.MatLike: Результирующее изображение
    """

    result = image_src.copy()
    if len(points) == 0:
        return result
    if relative:
        # Приводим относительные координаты в абсолютные
        points = (points * image_src.shape[:2][::-1])#.astype(np.int32)
    #points = points.astype(np.int32)
    # Отмечаем центры квадратов
    for p in points:
        # cv2.circle(vis, tuple(c), radius, color, thickness)
        cv2.line(result, (p.xi -size, p.yi), (p.xi+size, p.yi), color, thickness=thickness)
        cv2.line(result, (p.xi, p.yi-size), (p.xi, p.yi+size), color, thickness=thickness)
    return result

def draw_bboxes(title: str,
                image: cvtp.MatLike,
                norm_bboxes: [cmn.BoundingBox],
                is_relative: bool = True):
    """Отрисовка bboxes на изображении

    Args:
        title (str): Заголовок изображения
        image (cvtp.MatLike): Массив данных картинки
        bboxes ([BoundingBox]): Список bbox-ов
    """

    for norm_bbox in norm_bboxes:
        width = image.shape[1] if is_relative else 1
        height = image.shape[0] if is_relative else 1
        left = norm_bbox.left*width
        top = norm_bbox.top*height
        right = norm_bbox.right*width
        bottom = norm_bbox.bottom*height
        cv2.rectangle(image,
                      (int(round(left, 0)), int(round(top, 0))),
                      (int(round(right, 0)), int(round(bottom, 0))),
                      (0, 255, 0),
                      1)
    show_cv2_image(title=title, image=image)

def clear_dir(path: str):
    """Очистка папки (удаление/создание)

    Args:
        path (str): путь к папке
    """
    # Удаляем папку folder, если она уже есть
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f'Failed to delete {path}. Reason: {e}')

def create_dir(path: str):
    """Создание директории (если ее нет)

    Args:
        path (str): путь к папке
    """
    
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)

def copy_file(source_path: str, dest_path: str):
    """Копирование файла

    Args:
        source_path (str): Путь к файлу источника
        dest_path (str): Путь и имя файла назначения
    """
    
    shutil.copy2(source_path, dest_path)


def save_image(path: str, file_name:str, image: cvtp.MatLike):
    """Сохранение изображения в файл

    Args:
        path (str): Путь сохранения
        file_name (str): Имя файла
        image (cvtp.MatLike): Данные изображения
    """

    create_dir(path=path)
    cv2.imwrite(path+os.sep+file_name, image)

def text_to_file(path: str, file_name: str, text: str):
    """Сохранение текста в файл

    Args:
        path (str): Путь
        file_name (str): Имя файла
        text : Сохраняемый текст
    """

    create_dir(path=path)
    with open(path+os.sep+file_name, 'w') as file:
        file.write(text)