import { defineConfig } from "vite";

// No @preact/preset-vite: esbuild compiles the JSX directly, which keeps
// the dependency tree at exactly preact + preact-iso + vite + typescript
// (the spec's ceiling). The preset only adds HMR prefresh and babel.
export default defineConfig({
  base: "/v2/",
  esbuild: { jsx: "automatic", jsxImportSource: "preact" },
  build: { outDir: "dist", assetsDir: "assets" },
  server: {
    proxy: {
      // Dev-only convenience: `vite` against a running backend.
      "/api": "http://127.0.0.1:9293",
    },
  },
});
