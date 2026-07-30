---
description: Sanity-check the live llama-swap (:9292) and modelctl web (:9293) services.
---

Run the live checks from `docs/REVIEW-2026-07-29-moe-cache-integration.md` section 7:

```bash
curl -s localhost:9292/v1/models
curl -s -H "Authorization: Bearer $(cat ~/.local/share/modelctl/web_token)" localhost:9293/runtime
```

If a MoE model is running, also scrape its `/metrics` through the llama-swap upstream and report the `moe_cache_*` counters. Report what is loaded, any 5xx responses, and cache metrics if present. Never restart or kill these services unless explicitly asked. $ARGUMENTS
