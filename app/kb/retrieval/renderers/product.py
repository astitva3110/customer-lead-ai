"""Product document retrieval rendering."""

from app.kb.enums import SourceType
from app.kb.models.structured_content import DocumentContent
from app.kb.retrieval.renderers.common import clean_title, join_blocks


def _product_name(title: str, url: str) -> str:
    cleaned = clean_title(title)
    if "/products/" in url.lower():
        slug = url.rstrip("/").split("/")[-1]
        return slug.replace("-", " ").title()
    if cleaned.lower().endswith(".pdf"):
        return cleaned.replace(".pdf", "")
    return cleaned.split("\n")[0].strip(" -–")


def render_product(*, title: str, canonical_url: str, structured_content: DocumentContent) -> str:
    blocks = [f"Product:\n{_product_name(title, canonical_url)}"]

    def render_nodes(nodes) -> None:
        for node in nodes:
            node_type = node.type
            if node_type == "section":
                heading = (node.heading or "").strip()
                if heading:
                    blocks.append(heading)
                render_nodes(node.children)
            elif node_type == "heading":
                blocks.append(node.text.strip())
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

    if structured_content.pages:
        for page in structured_content.pages:
            blocks.append(f"Page {page.page_number}")
            render_nodes(page.blocks)
    else:
        render_nodes(structured_content.children)

    return join_blocks(blocks)
