# Hawker Agent Tool Contract Spec

This document is the project-level contract for model-facing tools and module
boundaries. Code, prompts, SOP generation, and tests must follow this contract.

## Architecture Boundaries

- `hawker_agent.browser`: browser lifecycle, navigation, DOM snapshots, clicks,
  form input, JavaScript execution, cookies, and browser-authenticated downloads.
- `hawker_agent.tools.http_tools`: explicit HTTP requests to known URLs. It must
  not infer requests from browser internals.
- `hawker_agent.tools.data_tools`: deterministic local data inspection and
  transformation, including JSON structure analysis.
- `hawker_agent.knowledge.observer`: post-run SOP generation from executed code,
  observations, and structured items. It must not collect extra browser state.
- `hawker_agent.agent.evaluator`: checks whether final answers are supported by
  recorded evidence. It must not accept unsupported self-claims.
- `hawker_agent.storage.exporter`: exports final summary and structured items
  without changing their semantics.

## Removed Network Capture Contract

Hawker does not expose browser network capture as a model tool.

- No `get_network_log()` tool.
- No `inspect_page(include=["network"])`.
- No injected Fetch/XHR monkey-patching.
- No hidden request sniffing in Observer or SOP generation.
- No prompt should recommend reading recent browser requests.

Reason: implicit network capture is timing-sensitive, browser-version-sensitive,
and frequently returns empty or misleading evidence. It also makes outputs hard
to reproduce because the true data source is not an explicit tool call.

## Replacement Data Access Contract

Agents must discover and replay data access explicitly:

- Use `search_web(...)` for web search. Treat upstream JSON as variable; call
  `analyze_json_structure(data)` before assuming field paths on unfamiliar
  payloads.
- Use `inspect_page(include=["dom"])` for visible page structure and actionable
  selectors.
- Use `js(...)` for page-local state, embedded JSON, script tags, and DOM-backed
  extraction. Returned values must be JSON-serializable.
- Use `fetch(...)` only with a known URL and explicit method/params/body/headers.
- Do not invent pagination parameters. Confirm `next` URL, cursor, offset,
  `hasMore`, or page semantics from the response, DOM/script state, docs, or a
  verified SOP before looping.
- Use `get_cookies(domain=...)` only when a known explicit request needs browser
  cookies.
- Use `append_items(...)` for structured deliverables and evidence.
- Use `final_answer(...)` for the user-facing completion summary only.

## Tool Output Rules

- `inspect_page` may return `dom`, `cookies`, and `selector`.
- `inspect_page` must reject unsupported dimensions such as `network` with a
  clear error.
- `fetch(parse="json")` returns parsed JSON.
- `fetch(parse="body")` returns the raw response body without status prefixes.
- `fetch(parse="text")` / `fetch(parse="raw")` are debugging modes and may
  include status metadata.
- `search_web(full=True)` returns raw upstream data plus schema analysis; callers
  must not assume a fixed provider shape.

## Coding Standards

- Keep tool wrappers thin. Tool behavior belongs in one module, not duplicated in
  prompts, Observer, and tests.
- Avoid compatibility paths that silently preserve removed behavior.
- Prefer explicit errors over empty placeholder structures.
- Do not add speculative abstractions. Add a new tool only when the contract and
  tests require it.
- Every model-visible tool must have tests for signature, output shape, and
  unsupported inputs.

## Testing Standards

- Contract tests must fail if removed network-capture names reappear in prompts
  or model-visible tool registries.
- Unit tests must cover unsupported `inspect_page(include=["network"])`.
- Observer tests must verify SOP generation does not call browser network tools.
- HTTP/data tests must cover `fetch(parse="body")` and
  `analyze_json_structure(...)` for unfamiliar JSON.
- End-to-end smoke tests should prove this flow: navigate or search, inspect DOM
  or page state, explicit fetch, structure analysis, append items, final answer.
