import cv2
import numpy as np
import cv2.typing as cvtp
import classes.common as cmn
import utils as utl
from scipy.stats import mode
from skimage.feature import canny
from imutils.perspective import four_point_transform
from imutils.object_detection import non_max_suppression 
from skimage.transform import hough_line, hough_line_peaks


class  Aligner:
    def __init__(self,MIN_WIDTH: int = 700, STEP: int = 10, THRESHOLD_1: float = 0.8, THRESHOLD_2: float = 0.85, MAX_DELTA: int = 5):
# Стартовая ширина масштабирования изображения при поиске маркеров
        self.MIN_WIDTH=int(MIN_WIDTH)
# Приращение ширины для следующей итерации
        self.STEP=int(STEP)
# Коэффициент соответствия черно-белого (левого верхнего) маркера оригиналу
        self.THRESHOLD_1=float(THRESHOLD_1)
# Коэффициент соответствия черных маркеров оригиналу
        self.THRESHOLD_2=float(THRESHOLD_2)
# Максимальная разница между соответствующими координатами центров маркеров,
# найденными на разных итерациях, чтобы считать что это один и тот же маркер
        self.MAX_DELTA=int(MAX_DELTA)

    def book_orientation_image(self,image: cvtp.MatLike)-> cvtp.MatLike:
        """Книжная ориентация изображения
            (если ширина больше высоты, поворот на 90 градусов)

        Args:
            image (cvtp.MatLike): Входное изображение

        Returns:
            cvtp.MatLike: Изображение в книжной ориентации
        """
        result = image.copy()
        if result.shape[1]>result.shape[0]:
            result = cv2.rotate(result, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        return result

    def rotate_image(self,image: cvtp.MatLike, angle: float)-> cvtp.MatLike:
        """Поворот изображения на заданный угол

        Args:
            image (cvtp.MatLike): Входное изображение
            angle (float): Угол поворота изображения

        Returns:
            cvtp.MatLike: Повернутое на заданный угол изображение
        """

        (h, w) = image.shape[: 2]
        center = (w // 2, h // 2)
        print('angle:',angle,type(angle))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC,
                                borderMode = cv2.BORDER_REPLICATE)
        return result

    def search_angle(self,image: cvtp.MatLike)-> float:
        """Поиск оптимального угла наклона изображения на основе преобразования Хафа

        Args:
            image (cvtp.MatLike): Входное изображение

        Returns:
            float: Найденный угол
        """

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = canny(image_gray)

        # преобразвание Хафа
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)

        angles = hough_line_peaks(h, theta, d)[1]
        most_common_angle = mode(np.around(angles, decimals=2))[0]
        skew_angle = np.rad2deg(most_common_angle - np.pi/2)

        return skew_angle

    def search_centers_of_markers(self,
            image_bw: cvtp.MatLike,
            width: int,
            threshold: float,
            marker: cvtp.MatLike)->[cmn.Point]:
        """Функция поиска координат центров областей, соответствующих маркеру

        Args:
            image_bw (cvtp.MatLike): Входное изображение
            width (int): Значение ширины, до которого выполняется масштабирование
                исходной картинки
            threshold (float): Пороговое значение соответствия маркеру
            marker (cvtp.MatLike): Изображение маркера, по которому выполнятеся
                поиск соответствий

        Returns:
            [cmn.Point]: _description_
        """

        ratio = width/min(image_bw.shape[:2])

        image_resize = cv2.resize(image_bw, None, fx=ratio, fy=ratio)
        # Passing the image to matchTemplate method 
        match = cv2.matchTemplate(image=image_resize,
                                  templ=marker,
                                  method=cv2.TM_CCOEFF_NORMED)

        # Select rectangles with confidence greater than threshold 
        (y_points, x_points) = np.where(match >= threshold) 

        # initialize our list of rectangles 
        boxes = list() 
        points = list()
        
        # marker dimensions 
        w, h = marker.shape[:2] 
        
        # loop over the starting (x, y)-coordinates again 
        for (x, y) in zip(x_points, y_points): 
            # update our list of rectangles 
            boxes.append((x, y, x + w, y + h)) 
            # img = utl.draw_crosses(image_resize, [cmn.Point(x+w/2, y+h/2)], relative=False)
            # utl.show_cv2_image('cross', img)

        # apply non-maxima suppression to the rectangles 
        # this will create a single bounding box 
        boxes = non_max_suppression(np.array(boxes)) 
        
        # loop over the final bounding boxes 
        for (x1, y1, x2, y2) in boxes: 
            points.append(cmn.Point(x=(x1+x2)/2/ratio, y=(y1+y2)/2/ratio))
        # print('width', width)
        # print('count points', len(points))
        return points

    def markers_searcher(self,image_bw: cvtp.MatLike,
                         marker: cvtp.MatLike,
                         begin_width = 700,
                         break_width = 3*700,
                         min_count: int = 1,
                         threshold: float =  0.8,
                         )->list[cmn.Point]:
        # Список количества найденных Point и самих Point
        result_points:list[list[int, cmn.Point]] = []
        min_width = 0
        is_continue = True
        is_found = False
        width = begin_width

        max_width = 0
        while is_continue:
            points = self.search_centers_of_markers(image_bw=image_bw,
                                               width=width,
                                               threshold=threshold,
                                               marker=marker)
            count = len(points)
            if is_found:
                if count == 0:
                    max_width = width - self.STEP
                    print(f'min_width = {min_width}, best_width = {(max_width+min_width)/2}, max_width = {max_width}')
                    is_continue = False# Закомментить, если не прекращать при 0

            else:
                if count > 0:
                    is_found = True
                    min_width = width

            for point in points:
                is_exists = False
                for value in result_points:
                    if abs(value[1].x - point.x) < self.MAX_DELTA\
                            and abs(value[1].y - point.y) < self.MAX_DELTA:
                        is_exists = True
                        value[0] += 1
                        value[1] = cmn.Point(x=(value[1].x + point.x)/2,
                                             y=(value[1].y + point.y)/2)
                        break
                if not is_exists:
                    count = len(result_points)
                    result_points.append([1, point])

            if len(result_points)<min_count:
                min_width = 0
                is_continue = True
                is_found = False
                
            width += self.STEP
            if width > break_width:
                max_width = width - self.STEP
                print(f'min_width = {min_width}, best_width = {(max_width+min_width)/2}, max_width = {max_width}')
                is_continue = False
                
        
        best_width = (min_width + max_width)//2
        # print ('min_width', min_width)
        # print ('best_width', best_width)
        # print ('max_width', max_width)
        print ('count', len(result_points))
        res:list[cmn.Point] = []
        if is_found:
            for value in sorted(result_points, key=lambda x:x[0], reverse=True):
                if len(res)<min_count:
                    res.append(value[1])
                else:
                    break

        return [res, best_width]

    def cropping_image_by_markers(self,image: cvtp.MatLike)->cvtp.MatLike:
        book_image = self.book_orientation_image(image=image)
        angle = self.search_angle(book_image)
        align_image = self.rotate_image(image=book_image, angle=angle[0])
        
        # Поворот ч/б изображения делаем отдельно от оригинала!
        # преобразование повернутого изображения к ч/б будет очень темным из-за
        # ярко-белого бордюра при достаточно больших углах поворота
        image_gray = cv2.cvtColor(book_image, cv2.COLOR_BGR2GRAY)
        # utl.show_cv2_image('Gray', image_gray)
        image_bw= cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # utl.show_cv2_image('B&W', image_bw)
        align_image_bw = self.rotate_image(image=image_bw, angle=angle[0])
        # utl.show_cv2_image('B&W rotated', align_image_bw)

        marker_1 = cv2.imread(
            'markers_images/bw_6_10_20.png')
        marker_1 = cv2.cvtColor(marker_1, cv2.COLOR_BGR2GRAY) 
        marker_1 = cv2.threshold(marker_1, 120, 255, cv2.THRESH_BINARY)[1]
        
        marker_2 = cv2.imread(
            'markers_images/b_10_20.png')
        marker_2 = cv2.cvtColor(marker_2, cv2.COLOR_BGR2GRAY) 
        marker_2 = cv2.threshold(marker_2, 120, 255, cv2.THRESH_BINARY)[1]
        
        # break_width = book_image.shape[1]
        markers_centers, best_width = self.markers_searcher(image_bw=align_image_bw,
                                           begin_width = self.MIN_WIDTH,
                                           break_width = 3*self.MIN_WIDTH,
                                           marker=marker_1,
                                           min_count=1,
                                           threshold=self.THRESHOLD_1                                       )
        if markers_centers[0].y > align_image_bw.shape[0]/2:
            is_rotate_180 = True
        else:
            is_rotate_180 = False
        markers_centers += self.markers_searcher(image_bw=align_image_bw,
                                           marker=marker_2,
                                           begin_width = 0.8*best_width,
                                           break_width = 1.2*best_width,
                                           min_count=3,
                                           threshold=self.THRESHOLD_2                                       )[0]
        img = utl.draw_crosses(align_image, markers_centers, size=20, thickness=2, relative=False)
    #    utl.show_cv2_image('Markers', img)
        c=[]
        for m in markers_centers:
            c.append([m.xi, m.yi])
        wraped_img = four_point_transform(img, np.array(c))
        if (is_rotate_180):
            wraped_img=cv2.rotate(wraped_img, rotateCode=cv2.ROTATE_180)
        return wraped_img
