# Количество знаков после запятой для округления координат для print()
PRINT_ROUND_DIGITS_COUNT = 1
# Количество знаков после запятой для округления координат для JSON
SAVE_ROUND_DIGITS_COUNT = 4

class Point:
    """Точка с координатами
    """
    def __init__(self, x:float, y:float):
        """Функция инициализации объекта

        Args:
            x (float): Координата x
            y (float): Координата y
        """
        
        self.x: float = x
        self.y: float = y
        self.xi: int = int(round(x, 0))
        self.yi: int = int(round(y, 0))

#____________________________________________________________________
class BoundingBox:
    """Класс ограничивающего объект прямоугольника
    """
    def __init__(self,
                left: float = 0.0,
                top: float = 0.0,
                right: float = 0.0,
                bottom: float = 0.0):
        """Функция инициализации объекта

        Args:
            left (float, optional): левая координата x. Defaults to 0.0.
            top (float, optional): верхняя координата y. Defaults to 0.0.
            right (float, optional): правая координата x. Defaults to 0.0.
            bottom (float, optional): нижняя координата y. Defaults to 0.0.
        """
        self.left: float = left
        self.top: float = top
        self.right: float = right
        self.bottom: float = bottom
        self.height: float = self.bottom-self.top
        self.width: float = self.right-self.left
        self.centre: Point = Point((self.left+self.right/2),
                                (self.top+self.bottom/2))
    def __str__(self):
        """Функция вывода данных экземпляра класса для print()
        """

        return f' left: {round(self.left, PRINT_ROUND_DIGITS_COUNT)},'\
            f' top: {round(self.top, PRINT_ROUND_DIGITS_COUNT)},'\
            f' right: {round(self.right, PRINT_ROUND_DIGITS_COUNT)},'\
            f' bottom: {round(self.bottom, PRINT_ROUND_DIGITS_COUNT)},'\
            f' width: {round(self.width, PRINT_ROUND_DIGITS_COUNT)},'\
            f' height: {round(self.height, PRINT_ROUND_DIGITS_COUNT)}'

    def get_JSON(self):
        return {
            'left': round(self.left, SAVE_ROUND_DIGITS_COUNT),
            'top': round(self.top, SAVE_ROUND_DIGITS_COUNT),
            'right': round(self.right, SAVE_ROUND_DIGITS_COUNT),
            'bottom': round(self.bottom, SAVE_ROUND_DIGITS_COUNT)
            }

    def get_bbox_abs(self, d_x:float=0.0, d_y:float=0.0,
                     padding_x: float = 0.0, padding_y: float = 0.0):
        """Функция получения bbox в абсолютных координатах с паддингом и смещением

        Args:
            d_x (float, optional): Смещение по x. Defaults to 0.0.
            d_y (float, optional): Смещение по y. Defaults to 0.0.
            padding_x (float, optional): Padding по x в %. Defaults to 0.0.
            padding_y (float, optional): Padding по y в %. Defaults to 0.0.

        Returns:
            BoundingBox: 
        """

        pad_x = self.width*padding_x/100.0
        pad_y = self.height*padding_y/100.0
        left = self.left - d_x - pad_x
        top = self.top - d_y - pad_y
        right = self.right - d_x + pad_x
        bottom = self.bottom - d_y + pad_y
        return BoundingBox(left=left,
                           top=top,
                           right=right,
                           bottom=bottom)
    
    def get_bbox_norm(self, d_x:float=0.0, d_y:float=0.0,
                     padding_x: float = 0.0, padding_y: float = 0.0,
                     div_x: float = 1.0, div_y: float = 1.0):
        """Функция получения bbox в относительных координатах с паддингом и смещением

        Args:
            d_x (float, optional): Смещение по x. Defaults to 0.0.
            d_y (float, optional): Смещение по y. Defaults to 0.0.
            padding_x (float, optional): Padding по x в %. Defaults to 0.0.
            padding_y (float, optional): Padding по y в %. Defaults to 0.0.
            div_x (float, optional): Делитель по x. Defaults to 1.0.
            div_y (float, optional): Делитель по y. Defaults to 1.0.

        Returns:
            BoundingBox: 
        """
        bbox_abs=self.get_bbox_abs(d_x=d_x, d_y=d_y, padding_x=padding_x, padding_y=padding_y)
        left = bbox_abs.left/div_x
        top = bbox_abs.top/div_y
        right = bbox_abs.right/div_x
        bottom = bbox_abs.bottom/div_y
        return BoundingBox(left=left,
                           top=top,
                           right=right,
                           bottom=bottom)

#____________________________________________________________________
class TextBox:
    """Класс строки для хранения текста с bbox
    """

    def __init__(self, text: str, bbox: BoundingBox):
        self.text: str = text
        self.bbox: BoundingBox = bbox

    def get_JSON(self):    
        return {
            'number': self.text,
            'bbox': self.bbox.get_JSON()
            }

#____________________________________________________________________