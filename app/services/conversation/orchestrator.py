from __future__ import annotations

import logging
import time
import uuid

from app.services.conversation.models import ConversationState
from app.interfaces.repositories.chat_trace_repository import ChatTraceRepository
from app.interfaces.repositories.conversation_repository import ConversationRepository
from app.helpers.conversation_trace import log_trace, snapshot_state

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    def __init__(
        self,
        graph,
        store: ConversationRepository,
        traces: ChatTraceRepository | None = None,
    ) -> None:
        self._graph = graph
        self._store = store
        self._traces = traces

    def handle(
        self,
        conversation_id: str | None,
        message: str,
        country: str | None = None,
        *,
        channel: str | None = None,
        origin: str | None = None,
        source: str | None = None,
        phone: str | None = None,
        user_name: str | None = None,
    ) -> ConversationState:
        from app.helpers.channel import apply_inbound_identity, resolve_conversation_id, resolve_request_source

        resolved_channel, resolved_origin = resolve_request_source(channel, origin, source)
        cid = resolve_conversation_id(conversation_id, resolved_channel or "web", phone)
        cid = cid or str(uuid.uuid4())
        state = self._store.get(cid) or ConversationState(conversation_id=cid)
        state.conversation_id = cid
        apply_inbound_identity(
            state,
            channel=resolved_channel,
            origin=resolved_origin,
            phone=phone,
            user_name=user_name,
        )
        state.user_message = message
        from app.config import settings
        from app.helpers.phone import normalize_region

        region = normalize_region(country) or normalize_region(state.session_country) or normalize_region(
            settings.default_country
        )
        if region:
            state.session_country = region
        state.response = ""
        state.sources = []
        state.error = ""
        state.guardrail_rejected = False
        state.query_rewritten = ""
        before = snapshot_state(state)
        state.trace = {"state_before": before}
        from app.services.diagnostics.recorder import TraceSession, tracing_enabled

        session = TraceSession.start(state, message)
        started = time.perf_counter()
        try:
            payload = self._invoke_graph(state)
            result = ConversationState.from_dict(payload)
            apply_inbound_identity(
                result,
                channel=resolved_channel,
                origin=resolved_origin,
                phone=phone,
                user_name=user_name,
            )
            result.trace = dict(result.trace or {})
            result.trace["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            session.finish(result)
            result.trace["trace_id"] = session.trace.trace_id
            observability = dict(getattr(session.trace, "observability", None) or {})
            if observability:
                result.trace["observability"] = {
                    "trace_id": session.trace.trace_id,
                    "langsmith_run_id": observability.get("langsmith_run_id"),
                    "token_usage": observability.get("token_usage"),
                }
            if tracing_enabled():
                from app.services.diagnostics.report import write_chat_trace

                write_chat_trace(session.trace)
            if settings.chat_debug_console:
                from app.services.diagnostics.console import print_chat_debug_console

                print_chat_debug_console(session.trace)
            self._persist_trace(session.trace)
            log_trace(result)
            self._store.save(result)
            return result
        except Exception as exc:
            from app.services.diagnostics.recorder import record_error

            record_error("graph", exc, recoverable=False, fallback_used=False)
            session.trace.latency["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
            if tracing_enabled():
                from app.services.diagnostics.report import write_chat_trace

                write_chat_trace(session.trace)
            if settings.chat_debug_console:
                from app.services.diagnostics.console import print_chat_debug_console

                print_chat_debug_console(session.trace)
            self._persist_trace(session.trace)
            raise
        finally:
            session.close()

    def _invoke_graph(self, state: ConversationState) -> dict:
        from app.services.diagnostics.langsmith_tracing import langsmith_enabled

        payload = state.to_dict()
        if not langsmith_enabled():
            return self._graph.invoke(payload)
        try:
            from langsmith import traceable

            @traceable(name="conversation_turn", run_type="chain")
            def _run(graph_payload: dict) -> dict:
                return self._graph.invoke(graph_payload)

            return _run(payload)
        except Exception:
            logger.exception("langsmith trace wrapper failed; invoking graph without wrapper")
            return self._graph.invoke(payload)

    def _persist_trace(self, trace) -> None:
        if self._traces is None:
            return
        try:
            self._traces.save(trace)
        except Exception:
            logger.exception("chat trace persist failed")
