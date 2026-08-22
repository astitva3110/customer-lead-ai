"""IPO prospectus retrieval rendering with section hierarchy."""

from app.kb.models.structured_content import DocumentContent
from app.kb.retrieval.renderers.common import clean_title, join_blocks, sanitize_block_text


def _is_prospectus(title: str, canonical_url: str) -> bool:
    combined = f"{title} {canonical_url}".lower()
    return "prospectus" in combined and "/investor/ipo/" in canonical_url.lower()


def render_prospectus(*, title: str, canonical_url: str, structured_content: DocumentContent) -> str:
    blocks = [f"Document:\n{clean_title(title)}"]
    current_section: str | None = None

    def append_section(name: str) -> None:
        nonlocal current_section
        cleaned = sanitize_block_text(name)
        if cleaned and cleaned != current_section:
            blocks.append(cleaned)
            current_section = cleaned

    def render_nodes(nodes) -> None:
        for node in nodes:
            node_type = node.type
            if node_type == "section":
                append_section(node.heading or "")
                render_nodes(node.children)
            elif node_type == "heading":
                append_section(node.text)
            elif node_type == "paragraph":
                cleaned = sanitize_block_text(node.text)
                if cleaned:
                    blocks.append(cleaned)
            elif node_type == "list":
                items = [sanitize_block_text(item) for item in node.items if sanitize_block_text(item)]
                if items:
                    blocks.append("\n".join(f"- {item}" for item in items))
            elif node_type == "table":
                table_lines = []
                if node.headers:
                    table_lines.append(" | ".join(node.headers))
                for row in node.rows:
                    table_lines.append(" | ".join(row))
                if table_lines:
                    blocks.append("\n".join(table_lines))

    if structured_content.pages:
        for page in structured_content.pages:
            blocks.append(f"Page:\n{page.page_number}")
            current_section = None
            render_nodes(page.blocks)
    else:
        render_nodes(structured_content.children)

    return join_blocks(blocks)


def maybe_render_prospectus(
    *,
    title: str,
    canonical_url: str,
    structured_content: DocumentContent,
) -> str | None:
    if _is_prospectus(title, canonical_url):
        return render_prospectus(title=title, canonical_url=canonical_url, structured_content=structured_content)
    return None
