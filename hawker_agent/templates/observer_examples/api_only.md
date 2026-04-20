# Example API Site — Scraping & Data Extraction

`https://api.example.org` — all public data, no auth required. **Never use the browser for this site.** One HTTP request returns structured JSON directly.

## Do this first

Use the JSON endpoint first and inspect the response shape before writing any parsing logic.

## Common workflows

### Search records (API)

```python
data = await http_json("https://api.example.org/search?q=transformer&limit=5")
items = data.get("results", [])
print(len(items), items[0]["title"] if items else None)
# Confirmed output (2026-04-20): 5 Transformer Foundations
```

### Fetch single record (API)

```python
record = await http_json("https://api.example.org/items/123")
print(record["id"], record["title"], record.get("updated_at"))
# Confirmed output (2026-04-20): 123 Transformer Foundations 2026-04-18
```

## API reference

| Field | Meaning |
| --- | --- |
| `q` | search query |
| `limit` | page size |
| `/items/{id}` | single record endpoint |

## Gotchas

- Do not request full DOM when JSON already contains the needed fields.
- Prefer direct item lookup over page scraping.
