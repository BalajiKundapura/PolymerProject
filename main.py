import sys
import os
import json

PROJECT_DIR = r"C:\Users\Balaji-Personal\Desktop\polymerProject"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import PDF_PATH, OUTPUT_DIR
from pdf_splitter import split_pdf_into_pages
from grobid_client import grobid_process_page, extract_body_text
from table_layers import get_table_bboxes
from text_layers import extract_text_ignoring_tables

def process_single_page(page_pdf, page_number):

    tei_xml = grobid_process_page(page_pdf)
    grobid_paragraphs = extract_body_text(tei_xml)

    table_bboxes = get_table_bboxes(page_pdf)

    pdf_text = extract_text_ignoring_tables(page_pdf, table_bboxes)
    pdf_paragraphs = [p.strip() for p in pdf_text.split("\n\n") if p.strip()]

    all_paragraphs = grobid_paragraphs + [p for p in pdf_paragraphs if p not in grobid_paragraphs]


    return {
        "page_number": page_number,
        "text": "\n\n".join(all_paragraphs),
        "tables": table_bboxes,
    }

def run_pipeline():
    page_files = split_pdf_into_pages(PDF_PATH)
    all_pages = []

    for i, page_pdf in enumerate(page_files, start=1):
        print(f"Processing page {i}")
        page_data = process_single_page(page_pdf, i)
        all_pages.append(page_data)

    output_path = os.path.join(OUTPUT_DIR, "multilayer_precise.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)

    print(f"Saved output to {output_path}")

if __name__ == "__main__":
    run_pipeline()
