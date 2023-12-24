import os
import json
import parsers
import utils as utl
import classes.common as cmn
import classes.bulletin as blt

#____________________________________________________________________
class Meeting:
    """Класс Голосования
    """

    def __init__(self, base_file_name: str):
        self.base_bulletin: blt.Bulletin = blt.Bulletin()
        parsers.ParserBasePDF.parse_base_pdf(self.base_bulletin, base_file_name, dpi = 100)
        self.number = self.base_bulletin.get_bulletin_number().text.split(sep='-')[-1]
        self.base_path = os.getcwd()+os.sep+'bulletins'+os.sep+self.number
        utl.clear_dir(self.base_path)
        utl.create_dir(self.base_path)
        utl.copy_file(source_path=base_file_name, dest_path=self.base_path+os.sep+'basePDF.pdf')

    def get_data(self)->[[int,
                          cmn.BoundingBox,
                          [cmn.TextBox],
                          [cmn.TextBox],
                          blt.Footer
                          ]]:
        return self.base_bulletin.get_data()

    def save_data(self, file_name: str = 'meeting.json'):
        utl.text_to_file(path=self.base_path,
                              file_name= file_name,
                              text=self.toJSON())
        for i, page in enumerate(self.base_bulletin.pages):
            utl.save_image(path=self.base_path+os.sep+'base_PDF_pages',
                           file_name='page_'+str(i+1)+'.png',
                           image=page.image)

    def get_JSON(self):
        return{
            'meeting': self.number,
            'base_path': self.base_path,
            'base_bulletin': self.base_bulletin.get_JSON()
            }

    def toJSON(self):
        return json.dumps(self.get_JSON(), 
            sort_keys=False, indent=4, ensure_ascii=False)

#____________________________________________________________________