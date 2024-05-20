from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple

import fitz_new as fitz  # PyMuPDF library

DATA_DIR = Path("./data/raw")


@dataclass
class PDF2Text:
    """Extracts plain text from a directory of PDF files, ignoring latex and code blocks."""

    # ! PYMUPDF KILLS THE KERNEL (Nov 28 2023). Check back later.

    pdf_dir: str = field(default=DATA_DIR)
    pdf_file: str = field(default=None)
    fonts_to_catch: Tuple[str] = field(
        default_factory=lambda: (
            "Helvetica",
            "CMR10",
            "LMRoman10",
            "SFRM1095",
            "LiberationSans",
            "Times",
            "Arial",
        )
    )  # RMarkdown, LaTeX (3), Chrominium, Google Docs (2)

    def load_pdf(self, pdf_path: str = None) -> str:
        """Extracts plain text from a PDF file using PyMuPDF.
        Args:
            pdf_path (str, optional):
                PDF file path
        Returns:
            plain_text (str): Plain text
        """
        if pdf_path is None:
            pdf_path = self.pdf_file

        doc = fitz.open(pdf_path)

        plain_text = "\n".join(
            [
                span["text"]
                for page in doc
                for block in page.get_text("dict")["blocks"]
                if block["type"] == 0  # 0 = block of text
                for line in block["lines"]
                for span in line["spans"]
                if (  # only extract spans with text of default styling
                    span["flags"] == 4
                    and len(span["text"]) > 5
                    and (span["font"].startswith(self.fonts_to_catch))
                )
            ]
        )

        return plain_text

    def load_pdf_dir(
        self, pdf_dir: str = None, return_dict: bool = False
    ) -> List[Dict[str, str]] | List[str]:
        """Construct a database of clean text from
        a directory of pdf files

        Args:
            pdf_dir (str, optional):
                Markdown directory path, default: self.pdf_dir
            return_dict (bool, optional):
                Whether to return a list of dictionaries with keys: text, source; by default, return a list of clean text
        Returns:
            data (List[Dict[str, str]] | List[str]):
                List of dictionaries with keys: text, source;
                or list of clean text (depending on return_dict)
        """
        if pdf_dir is None:
            pdf_dir = self.pdf_dir

        # get all PDF files in the directory and subdirectories
        files = list(Path(pdf_dir).rglob("*.pdf"))

        # load pdf files, and metadata if required
        data = []
        for file in files:
            text = self.load_pdf(file)
            if text.strip():
                if return_dict:
                    data.append({"text": text, "source": file.stem})
                else:
                    data.append(text)

        return data


if __name__ == "__main__":
    pdf2text = PDF2Text()
    docs = pdf2text.load_pdf_dir()
    print(len(docs))
