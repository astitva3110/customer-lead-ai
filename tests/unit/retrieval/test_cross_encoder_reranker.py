from app.config import settings
from app.services.retrieval.models import RetrievalCandidate
from app.providers.reranker.cross_encoder import CrossEncoderReranker
from app.providers.reranker.factory import create_reranker
from app.providers.reranker.lexical import LexicalOverlapReranker
from app.providers.reranker.passthrough import PassthroughReranker


def _candidate(chunk_id: str, text: str = "", **scores) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text or chunk_id,
        vector_score=scores.get("vector"),
        keyword_score=scores.get("keyword"),
    )


def test_create_reranker_providers() -> None:
    assert isinstance(create_reranker("lexical_overlap"), LexicalOverlapReranker)
    assert isinstance(create_reranker("passthrough"), PassthroughReranker)
    reranker = create_reranker("cross_encoder")
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.load_count == 0
    assert reranker.model_name == settings.reranker_model


def test_unknown_reranker_fails_clearly() -> None:
    try:
        create_reranker("mystery_ranker")
    except ValueError as exc:
        assert "Unknown reranker" in str(exc)
        assert "lexical_overlap" in str(exc)
        assert "cross_encoder" in str(exc)
    else:
        raise AssertionError("unknown provider should fail")


def test_cross_encoder_loads_once_and_reuses() -> None:
    loads: list[int] = []

    class Tok:
        def __call__(self, queries, texts, **kwargs):
            import torch

            del kwargs
            n = len(queries)
            assert n == len(texts)
            return {
                "input_ids": torch.ones((n, 4), dtype=torch.long),
                "attention_mask": torch.ones((n, 4), dtype=torch.long),
            }

    class Model:
        def __call__(self, **kwargs):
            import torch

            batch = next(iter(kwargs.values())).shape[0]
            out = type("Out", (), {})()
            out.logits = torch.zeros((batch, 1))
            return out

    def loader(model_name: str, device: str, revision: str | None):
        del model_name, device, revision
        loads.append(1)
        import torch

        return Tok(), Model(), torch.device("cpu")

    reranker = CrossEncoderReranker(model_name="mock-ce", loader=loader, device="cpu", batch_size=2)
    reranker.rerank("q", [_candidate("A", "aaa"), _candidate("B", "bbb")])
    reranker.rerank("q", [_candidate("C", "ccc")])
    assert reranker.load_count == 1
    assert loads == [1]


def test_cross_encoder_batches_query_text_pairs() -> None:
    seen: list[list[tuple[str, str]]] = []

    def score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        seen.append(list(pairs))
        return [0.2, 0.9, 0.5]

    reranker = CrossEncoderReranker(model_name="mock-ce", score_fn=score_fn, batch_size=16)
    candidates = [
        _candidate("A", "chunk A"),
        _candidate("B", "chunk B"),
        _candidate("C", "chunk C"),
    ]
    ranked = reranker.rerank("What is the price of Radius?", candidates)
    assert seen == [
        [
            ("What is the price of Radius?", "chunk A"),
            ("What is the price of Radius?", "chunk B"),
            ("What is the price of Radius?", "chunk C"),
        ]
    ]
    assert [item.chunk_id for item in ranked] == ["B", "C", "A"]


def test_cross_encoder_preserves_vector_and_keyword_scores() -> None:
    reranker = CrossEncoderReranker(
        model_name="mock-ce",
        score_fn=lambda pairs: [0.4, 0.95, 0.7],
    )
    ranked = reranker.rerank(
        "q",
        [
            _candidate("A", "a", vector=0.11, keyword=0.21),
            _candidate("B", "b", vector=0.12, keyword=0.22),
            _candidate("C", "c", vector=0.13, keyword=0.23),
        ],
    )
    by_id = {item.chunk_id: item for item in ranked}
    assert [item.chunk_id for item in ranked] == ["B", "C", "A"]
    assert by_id["A"].vector_score == 0.11
    assert by_id["A"].keyword_score == 0.21
    assert by_id["B"].vector_score == 0.12
    assert by_id["B"].keyword_score == 0.22


def test_cross_encoder_empty_candidates_skips_inference() -> None:
    called = {"n": 0}

    def score_fn(pairs):
        called["n"] += 1
        return [0.0] * len(pairs)

    reranker = CrossEncoderReranker(model_name="mock-ce", score_fn=score_fn)
    assert reranker.rerank("q", []) == []
    assert called["n"] == 0
    assert reranker.load_count == 0
