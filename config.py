import os

PDF_PATH = "pdfs/research_paper.pdf"

TEMP_PAGES_DIR = "temp_pages"
os.makedirs(TEMP_PAGES_DIR, exist_ok=True)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"