from __future__ import annotations

from typing import Protocol


class CrmLeadPort(Protocol):
    def create_lead(
        self,
        *,
        names: str,
        phone: str,
        source: str,
        email: str = "",
        problem: str = "",
        city: str = "null",
    ) -> None: ...
