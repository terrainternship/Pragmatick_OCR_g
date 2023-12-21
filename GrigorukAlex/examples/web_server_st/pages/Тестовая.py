import streamlit as st
import pandas as pd

uploaded_files = st.file_uploader("Загрузите базовый PDF собрания",type=['.pdf'], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    with open('ds/base_PDF/test.pdf', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.write("Файл загружен:", uploaded_file.name)
