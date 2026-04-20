# Example Browser-Required Site — Scraping & Data Extraction

`https://browser-only.example.org` — key data is rendered from live page state and does not have a stable public JSON endpoint. **Browser interaction is required.**

## Do this first

Navigate to the page, wait for the listing cards to render, and inspect the smallest useful DOM view before requesting a full snapshot.

## Common workflows

### Extract visible listing cards

```python
await nav("https://browser-only.example.org/listings", mode="summary")
cards_json = await js("""
(function(){
  const cards = Array.from(document.querySelectorAll(".listing-card"));
  return JSON.stringify(cards.slice(0, 5).map(card => ({
    title: card.querySelector(".title")?.innerText?.trim() || "",
    href: card.querySelector("a")?.href || "",
    price: card.querySelector(".price")?.innerText?.trim() || ""
  })));
})()
""")
print(cards_json[:160])
# Confirmed output (2026-04-20): [{"title":"Demo Listing","href":"https://browser-only.example.org/item/1","price":"$199"}]
```

### Paginate with the next-page control

```python
await click("a[rel='next']")
await dom_state(mode="diff")
cards_json = await js("""
(function(){
  return JSON.stringify(Array.from(document.querySelectorAll('.listing-card')).length);
})()
""")
print(cards_json)
# Confirmed output (2026-04-20): "25"
```

## Gotchas

- Do not invent an API path if the evidence only shows DOM-based extraction.
- Prefer `summary` or `diff` before asking for `full` DOM.
- Re-acquire selectors after navigation if the page structure changes.
