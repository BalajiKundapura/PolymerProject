import requests
try:
    from lxml import etree
    _HAVE_LXML = True
except Exception:
    import xml.etree.ElementTree as etree
    _HAVE_LXML = False

from config import GROBID_URL

def grobid_process_page(page_pdf):
    with open(page_pdf, "rb") as f:
        r = requests.post(GROBID_URL, files={"input": f})
        r.raise_for_status()
    return r.text

def extract_body_text(tei_xml):
    """Extract body paragraphs from TEI XML"""
    paragraphs = []

    if _HAVE_LXML:
        root = etree.fromstring(tei_xml.encode("utf-8"))
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        for div in root.xpath("//tei:div", namespaces=ns):
            for p in div.findall("tei:p", namespaces=ns):
                text = ''.join(p.itertext()).strip() if p is not None else ""
                if text:
                    paragraphs.append(text)
    else:
        root = etree.fromstring(tei_xml)
        ns_uri = "http://www.tei-c.org/ns/1.0"
        for div in root.findall('.//{' + ns_uri + '}div'):
            for p in div.findall('{' + ns_uri + '}p'):
                text = ''.join(p.itertext()).strip() if p is not None else ""
                if text:
                    paragraphs.append(text)

    return paragraphs
