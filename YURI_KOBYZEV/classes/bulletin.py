import utils as utl
import cv2.typing as cvtp
import classes.common as cmn

#____________________________________________________________________
class Footer:
    """Нижний колонтитул
    """
    
    # Размер bbox № страницы в % общей ширины bbox для № страницы и № бюллетеня
    PAGE_NUM_WIDTH_PERCENT = 10.0
    def __init__(self,
                 page_number: str,
                 bulletin_number: str,
                 bbox: cmn.BoundingBox):
        page_num_right = bbox.left + bbox.width*self.PAGE_NUM_WIDTH_PERCENT/100
        t_bbox = cmn.BoundingBox(left=bbox.left,
                                        top=bbox.top,
                                        right=page_num_right,
                                        bottom=bbox.bottom)
        self.page_number: cmn.TextBox = cmn.TextBox(text=page_number, bbox=t_bbox)
        t_bbox = cmn.BoundingBox(left=page_num_right,
                                        top=bbox.top,
                                        right=bbox.right,
                                        bottom=bbox.bottom)
        self.bulletin_number: cmn.TextBox = cmn.TextBox(text=bulletin_number, bbox=t_bbox)

    def get_JSON(self):
        return {
                'page_number': self.page_number.get_JSON(),
                'bulletin_number': self.bulletin_number.get_JSON()
                }

#____________________________________________________________________
class Page:
    'Страница'
    def __init__(self):
        self.bbox: cmn.BoundingBox # cmn.BoundingBox
         # Словарь id номеров строк голосования lineId:number
        self.__line_id_dict: dict[int, int] = {}
        self.question_lines: dict[int, cmn.TextBox] = {}
        self.footer: Footer
        self.markers: dict[int, cmn.TextBox] = {}
        self.image: cvtp.MatLike # пиксельное изображение страницы

    def set_question_line(self,
                          lineId: int,
                          question_num: str,
                          bbox: cmn.BoundingBox):
        if lineId not in self.__line_id_dict:
            self.__line_id_dict[lineId] = len(self.__line_id_dict)
            self.question_lines[self.__line_id_dict[lineId]] = cmn.TextBox(
                text=question_num,
                bbox=bbox)

    def set_marker(self, marker_name: str, bbox: cmn.BoundingBox):
        self.markers[len(self.markers)] = cmn.TextBox(text=marker_name, bbox=bbox)

    def get_JSON(self):
        json_text_m = {}
        for num, marker in self.markers.items():
            json_text_m[f'marker_{num+1}'] = marker.get_JSON()        
        json_text_q = {}
        for num, question_line in self.question_lines.items():
            json_text_q[f'line_{num+1}'] = question_line.get_JSON()
        return {
            'bbox': self.bbox.get_JSON(),
            'markers': json_text_m,
            'question_lines': json_text_q,
            'footer': self.footer.get_JSON()
            }

#____________________________________________________________________
class Bulletin:
    'Бюллетень голосования'
    
    def __init__(self):
        self.file_name: str
        self.bulletin_number: cmn.TextBox
        self.pages: list[Page] = []

    def set_bulletin_number(self, value: str, bbox: cmn.BoundingBox):
        self.bulletin_number = cmn.TextBox(text=value, bbox=bbox)

    def set_page_bbox(self, page_num: int, bbox: cmn.BoundingBox):
        self.pages[page_num].bbox=bbox

    def set_page_image(self, page_num: int, image: cvtp.MatLike):
        self.pages[page_num].image = image

    def set_page_question_line(self, page_num: int, lineId: int,
                        question_num:str, bbox: cmn.BoundingBox):
        self.pages[page_num].set_question_line(
            lineId=lineId,
            question_num=question_num,
            bbox = bbox)

    def set_page_footer(self,
                        page_num: int,
                        page_number: str,
                        bulletin_number: str,
                        bbox: cmn.BoundingBox):
        self.pages[page_num].footer = Footer(page_number=page_number,
                                        bulletin_number=bulletin_number,
                                        bbox = bbox)

    def get_bulletin_number(self):
        return self.bulletin_number

    def get_data(self)->[[int,
                          cmn.BoundingBox,
                          [cmn.TextBox],
                          [cmn.TextBox],
                          Footer]]:
        result = []
        for page_num, page in enumerate(self.pages):
            result.append([page_num, page.bbox, page.markers,
                           page.question_lines, page.footer])
        return result

    def draw_pages(self):
        """Отрисовываем изображениях страниц c нужными bboxes
        """

        for page_num, page in enumerate(self.pages):
            image = page.image.copy()
            width = page.bbox.width
            height = page.bbox.height
            bboxes = list()
            if page_num==0:
                utl.show_cv2_image(title="Base image", image=image)
                bboxes.append(self.bulletin_number.bbox.get_bbox_norm(
                    div_x=width,
                    div_y=height))
            for marker in page.markers.values():
                bboxes.append(marker.bbox.get_bbox_norm(
                    div_x=width,
                    div_y=height))
            for question_line in page.question_lines.values():
                bboxes.append(
                    question_line.bbox.get_bbox_norm(
                        padding_x=0.85,
                        padding_y=33.8,
                        div_x=width,
                        div_y=height))
            bboxes.append(page.footer.page_number.bbox.get_bbox_norm(
                    div_x=width,
                    div_y=height))
            bboxes.append(page.footer.bulletin_number.bbox.get_bbox_norm(
                    div_x=width,
                    div_y=height))
            utl.draw_bboxes(title='Page ' + str(page_num+1),
                            image=image,
                            norm_bboxes=bboxes)

    def get_JSON(self):
        json_text = {}
        for num, page in enumerate(self.pages):
            json_text[f'page_{num+1}'] = page.get_JSON()
        return {
            'file_name': self.file_name,
            'bulletin_number': self.bulletin_number.get_JSON(),
            'pages': json_text
            }

#____________________________________________________________________