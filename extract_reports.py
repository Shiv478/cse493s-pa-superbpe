from pathlib import Path

from pypdf import PdfReader


WORKSPACE = Path("/Users/shivvinodshankar/Documents/Codex/2026-05-28/files-mentioned-by-the-user-cse")
PDFS = [
    Path("/Users/shivvinodshankar/Downloads/CSE_493S_Final_Project_Version_1_Report.pdf"),
    Path("/Users/shivvinodshankar/Downloads/document_pdf.pdf"),
]


def main() -> None:
    output_dir = WORKSPACE / "extracted_reports"
    output_dir.mkdir(exist_ok=True)

    for pdf_path in PDFS:
        reader = PdfReader(str(pdf_path))
        chunks: list[str] = []
        for page_number, page in enumerate(reader.pages, 1):
            chunks.append(f"\n\n--- PAGE {page_number} ---\n")
            chunks.append(page.extract_text() or "")

        output_path = output_dir / f"{pdf_path.stem}.txt"
        output_path.write_text("".join(chunks), encoding="utf-8")
        print(f"{pdf_path.name}: {len(reader.pages)} pages -> {output_path}")


if __name__ == "__main__":
    main()
