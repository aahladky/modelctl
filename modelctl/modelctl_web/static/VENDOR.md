# Vendored assets

## htmx-2.0.4.min.js

- Upstream: https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js
- sha256: e209dda5c8235479f3166defc7750e1dbcd5a5c1808b7792fc2e6733768fb447
- Verified byte-identical against https://cdn.jsdelivr.net/npm/htmx.org@2.0.4/dist/htmx.min.js
  (two independent CDNs) on 2026-07-31.

Vendored rather than loaded from a CDN: the console is a local serving
tool that must work offline, and a CDN script tag with no integrity
attribute is arbitrary code execution inside the console's origin --
which holds a token that can set a profile's `binary` and load it.

To update: download both URLs at the new version, confirm the hashes
match each other, replace this file, and update the version in
`templates/base.html` and the hash above.
