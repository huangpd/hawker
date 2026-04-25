# Example Hybrid Site — Scraping & Data Extraction

`https://hybrid.example.org` — use browser to reveal page state, then replay the explicit JSON API.

## Do this first

Open the page, perform the minimal interaction, inspect DOM/script state with `js(...)`, then use `fetch(...)` once a concrete API URL is known.

## Common workflows

### Discover API hint from page state

```python
await nav("https://hybrid.example.org/search", mode="summary")
await fill_input(12, "llm agents")
await click_index(13)
state = await js("""() => Array.from(document.scripts).map(s => s.textContent).join("\\n").slice(0, 2000)""")
print("/api/search" in state)
# Confirmed output (2026-04-20): True
```

### Reuse API

```python
data = await fetch("https://hybrid.example.org/api/search?q=llm%20agents", parse="json")
items = data.get("items", [])
next_cursor = data.get("nextCursor")
print(len(items), bool(next_cursor))
# Confirmed output (2026-04-20): 20 True
```

## Gotchas

- Browser is only for discovering state, parameters, or auth context.
- Do not assume `page=1,2`; follow the response cursor/next URL/offset contract.
- Once the API pagination contract is known, do not paginate by clicking.
