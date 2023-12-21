import streamlit as st
import os
from glob import glob
from random import randint

from GrigorukAlex.parser_pdf.bboxes import parse

path_base_PDF = 'ds/base_PDF/'  # Директория с базовыми PDF-файлами


# Список имён доступных собраний
def get_names_meets(path_base_pdf: str):
    files = glob(path_base_pdf + '*.pdf')
    return sorted([os.path.split(file)[1].split('.')[0] for file in files])


names_meets = get_names_meets(path_base_PDF)
print(names_meets)
meet = st.sidebar.selectbox('Собрание', names_meets)

# Загрузка базового PDF
# https://discuss.streamlit.io/t/how-to-clear-uploaded-file-in-st-file-uploader/8029/3
if 'uploaded_file_key' not in st.session_state:
    st.session_state.uploaded_file_key = str(randint(1000, 100000000))
uploaded_file = st.sidebar.file_uploader("Загрузка базового PDF-файла",
                                         type=['.pdf'],
                                         key=st.session_state.uploaded_file_key,
                                         help="Выберите базовые PDF-файлы собраний, которые хотите добавить")

# Если загружен PDF-файл
if uploaded_file is not None:
    name_meet = str(uploaded_file.name)
    path_pdf = os.path.join(path_base_PDF + name_meet)
    with open(path_pdf, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.write("Файл загружен:", uploaded_file.name)

    parse(path_pdf,
          skip_num_tables=4,  # Кол-во таблиц, которые нужно пропустить на первой странице
          padding=(0.9, 0.65, 1.1, 0.6)
          # Отступ слева, сверху, справа, снизу от границы таблицы в % от ширины и высоты страницы
          )

    st.session_state.uploaded_file_key = str(randint(1000, 100000000))
    st.rerun()

tab1, tab2 = st.tabs(['Добавить собрание', 'Просмотр'])

with tab1:
    if meet is not None:
        st.write(meet)

        left_col, right_col = st.columns(2)

        with left_col:
            path_json = os.path.join(path_base_PDF, str(meet) + '.json')
            with open(path_json, 'r') as f:
                r = ''.join(f.readlines())
                st.json(r)

        with right_col:
            path_pages = os.path.join(path_base_PDF, str(meet))
            for img_file in os.listdir(path_pages):
                path_file = os.path.join(path_pages, img_file)
                st.image(path_file, width=800)

with tab2:
    st.write('tut')
