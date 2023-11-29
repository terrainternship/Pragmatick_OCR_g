from GrigorukAlex.parser_pdf.bboxes import parse


pdf_path = 'db/testPDF.pdf'
parse(pdf_path,
      skip_num_tables=4,  # Кол-во таблиц, которые нужно пропустить на первой странице
      padding=(0.9, 0.65, 1.1, 0.6)
      # Отступ слева, сверху, справа, снизу от границы таблицы в % от ширины и высоты страницы
      )
