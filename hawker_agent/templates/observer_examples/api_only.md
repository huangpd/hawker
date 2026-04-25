# Example API Site — Scraping & Data Extraction

`https://api.example.org` — public JSON API, no auth. **Never use the browser for list/detail data.**

## Do this first

Call the JSON endpoint and inspect the response shape before parsing fields.

## Common workflows

### Search records

```python
data = await fetch("https://api.example.org/search?q=transformer&limit=5", parse="json")
items = data.get("results", [])
print(len(items), items[0]["title"] if items else None)
# Confirmed output (2026-04-20): 5 Transformer Foundations
```

### Fetch detail

```python
record = await fetch("https://api.example.org/items/123", parse="json")
print(record["id"], record["title"])
# Confirmed output (2026-04-20): 123 Transformer Foundations
```

## Gotchas

- Do not request DOM when JSON already has the fields.
- Prefer direct detail endpoints over browser scraping.
