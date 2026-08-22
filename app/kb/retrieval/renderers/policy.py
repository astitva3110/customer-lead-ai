"""Policy document retrieval rendering."""

from app.kb.models.structured_content import DocumentContent
from app.kb.retrieval.renderers.common import clean_title, join_blocks


def render_policy(*, title: str, structured_content: DocumentContent) -> str:
    blocks = [clean_title(title)]

    def render_nodes(nodes, *, indent: str = "") -> None:
        for node in nodes:
            node_type = node.type
            if node_type == "section":
                heading = (node.heading or "").strip()
                if heading and heading.lower() not in blocks[-1].lower():
                    blocks.append(f"{indent}{heading}")
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

    render_nodes(structured_content.children)
    return join_blocks(blocks)
