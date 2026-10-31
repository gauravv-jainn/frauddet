# report_kit

Facts, tables, figures and sources for the college report book. This is not the report itself. Start with `REPORT_FACTS.md`.

| File or folder | What it is |
|---|---|
| `REPORT_FACTS.md` | 17 sections: title/abstract/stack, problem, base paper, literature, dataset, architecture, methodology, results, testing, timeline, screenshots, limitations/ethics, 25 viva questions, open questions, inconsistencies, template cross-reference, self-check log |
| `references.md` | IEEE references: verified, web pages (reachable only), and UNVERIFIED items |
| `code_snippets.md` | 15 verbatim excerpts (at most 25 lines each) with path and line range |
| `glossary.md`, `figure_table_index.md` | Terms; suggested figure and table numbers |
| `tables/*.csv` | Every results and literature table (one CSV each) |
| `figures/` | 5 generated diagrams (PNG) and 12 screenshots; sources in `figures/diagrams_src/` |
| `_data/` | Raw evidence: pytest output, CI runs, SHAP example output, extra measurements, reference check logs |
| `scripts/` | Code that regenerates everything from `fraudlens/artifacts/` (read-only on `fraudlens/`) |

Regenerate (from the repository root, nothing is retrained):

```
fraudlens/.venv/Scripts/python.exe report_kit/scripts/extract_results.py
fraudlens/.venv/Scripts/python.exe report_kit/scripts/extra_facts.py
fraudlens/.venv/Scripts/python.exe report_kit/scripts/verify_refs2.py
fraudlens/.venv/Scripts/python.exe report_kit/scripts/build_tables.py
fraudlens/.venv/Scripts/python.exe report_kit/scripts/make_snippets.py
fraudlens/.venv/Scripts/python.exe report_kit/figures/diagrams_src/make_matplotlib_figures.py
fraudlens/.venv/Scripts/python.exe report_kit/scripts/build_facts.py
```

`verify_refs.py` (first, search-based pass) and `verify_refs2.py` (direct DOI check) need internet access; the Graphviz diagrams were rendered with `dot -Tpng -Gdpi=170 file.dot -o file.png`.
