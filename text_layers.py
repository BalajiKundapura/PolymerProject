import pdfplumber

def extract_text_ignoring_tables(page_pdf, table_bboxes):
    with pdfplumber.open(page_pdf) as pdf:
        page = pdf.pages[0]
        words = page.extract_words()
        filtered_words = []

        for w in words:
            x0, top, x1, bottom = w['x0'], w['top'], w['x1'], w['bottom']
            inside_table = False
            for bbox in table_bboxes:
                bx0, by0, bx1, by1 = bbox
                if x0 >= bx0 and x1 <= bx1 and top >= by0 and bottom <= by1:
                    inside_table = True
                    break
            if not inside_table:
                filtered_words.append(w)

        lines = {}
        for w in filtered_words:
            lines.setdefault(int(w['top']), []).append(w['text'])
        page_text = "\n".join([" ".join(words) for words in lines.values()])
        return page_text
