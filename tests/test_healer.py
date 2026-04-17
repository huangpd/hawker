from __future__ import annotations

import pytest

from hawker_agent.agent.healer import estimate_change_ratio
from hawker_agent.models.state import CodeAgentState


class TestHealer:
    def test_change_ratio_small_for_local_fix(self) -> None:
        original = "items = []\nobserve(str(len(itemz)))"
        candidate = "items = []\nobserve(str(len(items)))"
        assert estimate_change_ratio(original, candidate) < 0.55

    def test_change_ratio_large_for_full_rewrite(self) -> None:
        original = "items = []\nobserve(str(len(itemz)))"
        candidate = (
            "data = await fetch(url, parse='json')\n"
            "rows = data['items']\n"
            "await append_items(rows)\n"
            "await final_answer('done')"
        )
        assert estimate_change_ratio(original, candidate) > 0.55

    @pytest.mark.asyncio
    async def test_healing_appends_accepted_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hawker_agent.agent import healer as healer_mod
        from hawker_agent.agent.namespace import HawkerNamespace

        class FakeResponse:
            text = "```python\nitems = []\nobserve(str(len(items)))\n```"
            input_tokens = 1
            output_tokens = 1
            cached_tokens = 0
            total_tokens = 2
            cost = 0.0

        class FakeClient:
            def __init__(self, cfg): ...

            async def complete_with_model(self, *args, **kwargs):
                return FakeResponse()

        class FakeSettings:
            healer_enabled = True
            small_model_name = "mini"
            healer_reasoning_effort = ""
            healer_max_attempts = 1

        monkeypatch.setattr(healer_mod, "get_settings", lambda: FakeSettings())
        monkeypatch.setattr(healer_mod, "LLMClient", FakeClient)

        state = CodeAgentState()
        namespace = HawkerNamespace({}, "/tmp/run")
        healed = await healer_mod.try_heal_code(
            code="items = []\nobserve(str(len(itemz)))",
            error="[执行错误]\nNameError: name 'itemz' is not defined",
            namespace=namespace,
            state=state,
        )

        assert healed is not None
        assert state.healing_records[-1]["status"] == "accepted"
