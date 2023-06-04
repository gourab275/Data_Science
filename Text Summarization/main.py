import streamlit as st
from txtai.pipeline import Summary , Textractor
from PyPDF2 import PdfReader

st.set_page_config(layout='wide')

@st.cache_resource
def text_summary (text, maxlength = None):
    summary = Summary()
    text = (text)
    result = summary(text)
    return  result

def extract_text_from_pdf(file_path):
    with open(file_path , 'rb') as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return  text







choice = st.sidebar.selectbox('Select your choice',["Summarize text", "Summarize Document"])

if choice == 'Summarize text':
    st.subheader('Summarize text')
    input_text = st.text_area('Enter Your Text Here')
    if input_text is not None:
        if st.button("Summarize Text"):
            col1 ,col2 = st.columns([1,1])
            with col1:
                st.markdown('--- Your Input Text ---')
                st.info(input_text)
            with col2:
                st.markdown('--Summary Result--')
                result = text_summary(input_text)
                st.success(result)





elif choice == 'Summarize Document':
    st.subheader("Summarize Documents")
    input_file = st.file_uploader('Upload Your Document Here', type=['pdf'])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf",'wb') as f:
                f.write(input_file.getbuffer())
            col1 , col2 = st.columns([1,1])
            with col1 :
                st.markdown("---Extracted text ---")
                extracted_text = extract_text_from_pdf('doc_file.pdf')
                st.info(extracted_text)

            with col2:
                result = extract_text_from_pdf('doc_file.pdf')
                st.markdown("--Summarize Document--")
                summary_result = text_summary(result)
                st.success(summary_result)




