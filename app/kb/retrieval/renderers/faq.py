"""FAQ document retrieval rendering."""

import re

from app.kb.models.structured_content import DocumentContent
from app.kb.retrieval.renderers.common import clean_title, join_blocks

FAQ_INLINE = re.compile(r"Q(\d+)\.\s+(.+?)(?=Q\d+\.|$)", re.IGNORECASE | re.DOTALL)


def _append_qa(blocks: list[str], question: str, answer: str) -> None:
    question = question.strip().rstrip("?") + "?"
    answer = answer.strip()
    if question and answer:
        blocks.append(f"Question:\n{question}\n\nAnswer:\n{answer}")


def render_faq(*, title: str, structured_content: DocumentContent) -> str:
    blocks: list[str] = []
    pending_question: str | None = None

    def flush_pending(answer: str) -> None:
        nonlocal pending_question
        if pending_question:
            _append_qa(blocks, pending_question, answer)
            pending_question = None

    def render_nodes(nodes) -> None:
        nonlocal pending_question
        for node in nodes:
            node_type = node.type
            if node_type == "section":
                render_nodes(node.children)
            elif node_type == "heading":
                if node.text.strip().endswith("?"):
                    pending_question = node.text.strip()
                else:
                    flush_pending(node.text)
            elif node_type == "paragraph":
                text = node.text.strip()
                inline_matches = list(FAQ_INLINE.finditer(text))
                if inline_matches:
                    for match in inline_matches:
                        _append_qa(blocks, match.group(2), "")
                    continue
                if pending_question:
                    flush_pending(text)
                elif text.upper().startswith("Q") and "?" in text[:120]:
                    parts = text.split("?", 1)
                    _append_qa(blocks, parts[0] + "?", parts[1] if len(parts) > 1 else "")
                else:
                    blocks.append(text)
            elif node_type == "list":
                body = "\n".join(f"- {item.strip()}" for item in node.items if item.strip())
                if pending_question:
                    flush_pending(body)
                elif body:
                    blocks.append(body)

    render_nodes(structured_content.children)

    if structured_content.pages:
        for page in structured_content.pages:
            render_nodes(page.blocks)

    if not blocks:
        blocks.append(clean_title(title))

    return join_blocks(blocks)
