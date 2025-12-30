from PyPDF2 import PdfReader, PdfWriter
from config import TEMP_PAGES_DIR
import os

def split_pdf_into_pages(pdf_path):
    reader = PdfReader(pdf_path)
    page_files = []

    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)

        page_path = os.path.join(TEMP_PAGES_DIR, f"page_{i}.pdf")
        with open(page_path, "wb") as f:
            writer.write(f)

        page_files.append(page_path)

    return page_files
