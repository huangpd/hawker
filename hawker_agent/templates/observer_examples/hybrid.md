# Example Hybrid Site — Scraping & Data Extraction

`https://hybrid.example.org` — mixed browser and API workflow. Use the browser only to establish page state, then switch to the underlying JSON/XHR path as soon as it becomes visible.

## Do this first

Open the page once, trigger the minimal interaction that reveals the data request, then inspect `get_network_log()` for the actual API path.

## Common workflows

### Reveal the backing API (browser -> network)

```python
await nav("https://hybrid.example.org/search", mode="summary")
await dom_state(mode="full")
await fill_input(12, "llm agents")
await click_index(13)
logs = await get_network_log(content_type_contains="json", max_entries=10)
print(logs["summary"]["likely_data_api"])
# Confirmed output (2026-04-20): [{'url': 'https://hybrid.example.org/api/search?q=llm%20agents', ...}]
```

### Reuse the API directly

```python
data = await http_json("https://hybrid.example.org/api/search?q=llm%20agents&page=1")
items = data.get("items", [])
print(len(items), items[0]["title"] if items else None)
# Confirmed output (2026-04-20): 20 LLM Agents in Production
```

## API reference

| Field | Meaning |
| --- | --- |
| `q` | search query |
| `page` | 1-indexed page number |

## Gotchas

- The browser is only for discovering the request path and required parameters.
- Once the API path is known, do not keep clicking through pagination in the browser.
