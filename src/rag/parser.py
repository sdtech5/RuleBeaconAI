import re

from langchain_core.documents import Document


PART_PATTERN = re.compile(
    r"^\**\s*PART\s+(I{1,3}|IV)\s*\**$",
    re.IGNORECASE,
)

ITEM_PATTERN = re.compile(
    r"^\**\s*Item[\s\u00A0]+(\d+[A-Z]?)\.\s*(.*?)\s*\**$",
    re.IGNORECASE,
)


def parse_document(document: Document) -> list[Document]:
    """
    Parse an SEC 10-K document into structurally aware sections.

    Each returned Document represents a continuous section of the
    original document and carries its structural metadata.
    """

    current_part = None
    current_item = None
    current_section = None

    sections = []
    current_lines = []

    def save_section():
        if not current_lines:
            return

        text = "\n".join(current_lines).strip()

        if not text:
            return

        metadata = document.metadata.copy()

        metadata["document_type"] = "SEC 10-K"
        metadata["part"] = current_part
        metadata["item"] = current_item
        metadata["section"] = current_section

        sections.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    for line in document.page_content.splitlines():

        stripped = line.strip()

        # Normalize Markdown table formatting and non-breaking spaces
        normalized = stripped.replace("|", " ").strip()
        normalized = normalized.replace("\u00A0", " ")

        part_match = PART_PATTERN.match(normalized)

        if part_match:
            save_section()

            current_lines = []
            current_part = f"Part {part_match.group(1).upper()}"
            current_item = None
            current_section = None

            current_lines.append(line)
            continue

        item_match = ITEM_PATTERN.match(normalized)

        if item_match:
            save_section()

            current_lines = []
            current_item = f"Item {item_match.group(1).upper()}"
            current_section = item_match.group(2).strip()

            current_lines.append(line)
            continue

        current_lines.append(line)

    save_section()

    return sections