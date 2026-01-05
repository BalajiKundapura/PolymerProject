import requests
from lxml import etree
from pathlib import Path
import csv
#docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.1

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

NS = {
    "tei": "http://www.tei-c.org/ns/1.0"
}

def extract_text_and_tables_from_pdf(pdf_path, text_out, tables_dir):
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(pdf_path, "rb") as pdf_file:
        response = requests.post(
            GROBID_URL,
            files={"input": pdf_file},
            data={
                "consolidateHeader": "1",
                "consolidateCitations": "1",
                "teiCoordinates": "false"
            },
            timeout=60
        )

    if response.status_code != 200:
        raise RuntimeError(f"GROBID failed: {response.text}")

    root = etree.fromstring(response.content)

    text_nodes = root.xpath(
        "//tei:body//text()[not(ancestor::tei:table)]",
        namespaces=NS
    )

    clean_text = "\n".join(t.strip() for t in text_nodes if t.strip())
    Path(text_out).write_text(clean_text, encoding="utf-8")

    tables = root.xpath("//tei:table", namespaces=NS)

    for idx, table in enumerate(tables, start=1):
        rows = table.xpath(".//tei:row", namespaces=NS)

        table_data = []
        max_cols = 0

        for row in rows:
            cells = row.xpath(".//tei:cell", namespaces=NS)
            row_data = []

            for cell in cells:
                cell_text = " ".join(
                    t.strip() for t in cell.xpath(".//text()", namespaces=NS) if t.strip()
                )
                row_data.append(cell_text)

            max_cols = max(max_cols, len(row_data))
            table_data.append(row_data)

        for row in table_data:
            row.extend([""] * (max_cols - len(row)))

        csv_path = tables_dir / f"table_{idx}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(table_data)

    print(f"✅ Text saved to: {text_out}")
    print(f"✅ {len(tables)} tables saved as CSV in: {tables_dir}")


if __name__ == "__main__":
    extract_text_and_tables_from_pdf(
        pdf_path=r"C:\Users\Balaji-Personal\Desktop\PolymerProject-1\inputs\samplePaper2.pdf",
        text_out="rawData/paper.txt",
        tables_dir="rawData/tables"
    )
