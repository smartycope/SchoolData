import streamlit as st
from streamlit import session_state as ss
import fitz  # PyMuPDF
import re
import ezregex
from docx import Document
import bs4

files = st.file_uploader('Upload a file or files to scrape', ['txt', 'pdf', 'html', 'docx'], True)

if 'keywords' not in ss:
    ss.keywords = [
        "enrollment cap",
        "refinance",
        "expansion",
        "startup",
        "new schools",
        "merger",
        "surrender",
        "charter proposal",
    ]

with st.sidebar:
    keywords = st.data_editor(ss.keywords)
    st.text_input('Add a keyword', on_change=lambda: ss.keywords.append(st.session_state['keyword']), key='keyword')
    context_chars = st.number_input('Context characters', value=100, min_value=0, max_value=300, step=1)


def parse_text(file):
    rtn = []
    with open(file) as f:
        text = f.read()
        if (m := re.search('|'.join(ss.keywords), text, re.IGNORECASE)):
            rtn.append({
                'file': file,
                'page': 0,
                'text': text,
                'match': m,
            })
    return rtn


def parse_docx(file):
    rtn = []
    doc = Document(file)
    text = "\n".join(p.text for p in doc.paragraphs)
    if (m := re.search('|'.join(ss.keywords), text, re.IGNORECASE)):
        rtn.append({
            'file': file,
            'page': 0,
            'text': text,
            'match': m,
        })
    return rtn


def parse_pdf(file):
    rtn = []
    doc = fitz.open(file)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if (m := re.search('|'.join(ss.keywords), text, re.IGNORECASE)):
            rtn.append({
                'file': file,
                'page': page_num,
                'text': text,
                'match': m,
            })
    return rtn


def parse_html(file):
    rtn = []
    with open(file) as f:
        text = bs4.BeautifulSoup(f.read(), 'html.parser').get_text()
        if (m := re.search('|'.join(ss.keywords), text, re.IGNORECASE)):
            rtn.append({
                'file': file,
                'page': 0,
                'text': text,
                'match': m,
            })
    return rtn

def search(file):
    match file.type:
        case 'text/plain':
            return parse_text(file)
        case "application/msword":
            return parse_docx(file)
        case "application/pdf":
            return parse_pdf(file)
        case 'text/html':
            return parse_html(file)

if not files:
    st.info("No files selected")
    st.stop()
matches = []
for i in files:
    if (found := search(i)):
        matches += found

st.success(f"Found {len(matches)} matches")

for i in matches:
    st.write(f'In file `{i["file"].name}`, on page {i["page"] + 1}, found the text `{i["match"].group()}`')
    with st.expander('Context'):
        st.code(i['text'][max(0, i['match'].start() - context_chars):min(len(i['text']), i['match'].end() + context_chars)], language='text')


