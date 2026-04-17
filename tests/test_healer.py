from __future__ import annotations

from hawker_agent.agent.healer import estimate_change_ratio


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
