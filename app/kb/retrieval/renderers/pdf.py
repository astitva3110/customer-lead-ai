"""PDF document retrieval rendering."""

from app.kb.models.structured_content import DocumentContent
from app.kb.retrieval.renderers.common import clean_title, join_blocks


def render_pdf(*, title: str, structured_content: DocumentContent) -> str:
    blocks = [f"Document:\n{clean_title(title)}"]
    current_section: str | None = None

    def append_section(name: str) -> None:
        nonlocal current_section
        cleaned = name.strip()
        if cleaned and cleaned != current_section:
            blocks.append(cleaned)
            current_section = cleaned

    def render_blocks(nodes) -> None:
        for node in nodes:
            node_type = node.type
            if node_type == "heading":
                append_section(node.text)
            elif node_type == "paragraph":
                blocks.append(node.text.strip())
            elif node_type == "list":
                blocks.append("\n".join(f"- {item.strip()}" for item in node.items if item.strip()))
            elif node_type == "table":
                table_lines = []
                if node.headers:
                    table_lines.append(" | ".join(node.headers))
                for row in node.rows:
                    table_lines.append(" | ".join(row))
                blocks.append("\n".join(table_lines))
            elif node_type == "section":
                append_section(node.heading or "")
                render_blocks(node.children)

    if structured_content.pages:
        for page in structured_content.pages:
            blocks.append(f"Page:\n{page.page_number}")
            current_section = None
            render_blocks(page.blocks)
    else:
        render_blocks(structured_content.children)

    return join_blocks(blocks)
