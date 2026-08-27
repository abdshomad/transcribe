# Feature: Multi-Model Storage & Comparison Engine

- **Persistence**: Independent run storage per `(source_name, model)` in SQLite (`src/transcribe/history.py`).
- **Diff Engine**: `difflib.SequenceMatcher` token diffs with normalized case/punctuation.
- **Visual Presentation**: Color-coded word diffs (Green additions, Red deletions, Amber substitutions) & timeline view.
- **Benchmark Metrics**: Similarity score %, processing speedup ($X\times$), word count delta, and speaker counts.
- **Quick Runner**: 1-click model runner chips (`Tiny`, `Base`, `Small`, `Medium`, `Large-v3`) in web banner.
