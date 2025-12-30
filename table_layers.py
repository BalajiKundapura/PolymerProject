import camelot

def get_table_bboxes(page_pdf):
    bboxes = []

    try:
        tables = camelot.read_pdf(page_pdf, pages="1", flavor="lattice")
        bboxes.extend([t._bbox for t in tables])
    except:
        pass

    try:
        tables = camelot.read_pdf(page_pdf, pages="1", flavor="stream")
        bboxes.extend([t._bbox for t in tables])
    except:
        pass

    return bboxes
