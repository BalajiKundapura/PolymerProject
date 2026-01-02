from chemdataextractor import Document
from chemdataextractor.model import (
    BaseModel,
    StringType,
    ModelType,
    ListType,
    Compound
)
from chemdataextractor.parse import (
    I,
    R,
    Optional,
    BaseSentenceParser,
    AutoSentenceParser
)

# =========================================================
# MODEL
# =========================================================

class SolventRatio(BaseModel):
    value = StringType()

    specifier = StringType(
        parse_expression=(
            I('solvent') + I('ratio')
            | I('solvent') + I('mixture')
            | I('mixture')
        )
    )

    compound = ModelType(Compound, contextual=True)

    parsers = [AutoSentenceParser()]


Compound.solvent_ratios = ListType(ModelType(SolventRatio))


# =========================================================
# PARSER
# =========================================================

prefix = (
    I('solvent') +
    Optional(I('mixture')) +
    I('ratio') +
    Optional(I('of'))
).hide()

ratio_value = R(
    r'\d+(\.\d+)?\s*[:/]\s*\d+(\.\d+)?'
)('value')

root = prefix + ratio_value


class SolventRatioParser(BaseSentenceParser):
    root = root

    def interpret(self, result, start, end):
        yield SolventRatio(
            value=result.xpath('./value/text()')[0]
        )


SolventRatio.parsers.append(SolventRatioParser())


# =========================================================
# FILE-BASED INPUT
# =========================================================

def load_text_from_file(path: str) -> str:
    """Reads a text file and returns its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":

    TEXT_FILE_PATH = r"C:\Users\Balaji-Personal\Desktop\PolymerProject-1\rawData\paper.txt"
    text = load_text_from_file(TEXT_FILE_PATH)

    doc = Document(text)

    print("\n=== EXTRACTED SOLVENT RATIOS ===")
    for record in doc.records:
        if isinstance(record, SolventRatio):
            print(f"Ratio: {record.value}")
