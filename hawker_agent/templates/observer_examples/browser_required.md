# Example Browser-Required Site — Scraping & Data Extraction

`https://browser-only.example.org` — data is rendered from live page state. **Browser interaction is required.**

## Do this first

Navigate, wait for cards, then use the smallest useful DOM/JS extraction.

## Common workflows

### Extract visible cards

```python
await nav("https://browser-only.example.org/listings", mode="summary")
cards = await js("""() => Array.from(document.querySelectorAll(".listing-card")).slice(0, 5).map(card => ({
  title: card.querySelector(".title")?.innerText?.trim() || "",
  href: card.querySelector("a")?.href || "",
  price: card.querySelector(".price")?.innerText?.trim() || ""
}))""")
print(len(cards), cards[0]["title"] if cards else None)
# Confirmed output (2026-04-20): 5 Demo Listing
```

### Paginate

```python
await click("a[rel='next']")
await dom_state(mode="diff")
# Confirmed output (2026-04-20): page changed, listing cards refreshed
```

## Gotchas

- Do not invent an API path if evidence only supports DOM extraction.
- Re-acquire selectors after navigation or major DOM refresh.
