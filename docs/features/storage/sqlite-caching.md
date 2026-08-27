# Feature: High-Speed SQLite Storage & Client Caching

- **Database Optimization**: SQLite Write-Ahead Logging (`PRAGMA journal_mode = WAL;`) with normal sync (`PRAGMA synchronous = NORMAL;`).
- **Indexes**: Compound index on `created_at DESC` and `source_name` for fast sorting and grouping.
- **Startup Caching**: One-time schema initialization at startup without per-query overhead.
- **Client Cache**: Background prefetching & Stale-While-Revalidate caching for 0ms history drawer rendering.
