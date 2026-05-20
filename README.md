# Japanese Song Lyrics Translator

Learn Japanese through music. Paste a YouTube URL to get furigana readings, word-by-word translations, and karaoke-style lyric highlighting.

## Features

- **Furigana & Romaji** — every kanji annotated with its reading
- **AI-powered analysis** — per-word breakdown, grammar points, JLPT levels (via Groq / Llama 3.3)
- **Dictionary lookups** — Jotoba + Jisho fallback when no AI key is set
- **Anki integration** — one-click card export with styled front/back HTML
- **Karaoke player** — YouTube embed with lyrics panel *(timing sync coming soon)*
- **SQLite + Prisma** — lyrics and analysis cached locally

## Monorepo Structure

```
japanese-song-lyrics-translator/
├── apps/
│   └── web/                        # Next.js 14 app (App Router)
│       ├── src/app/                # Pages & API routes
│       ├── src/components/         # React components + shadcn/ui
│       ├── src/lib/                # Prisma client, utilities
│       └── prisma/schema.prisma    # SQLite schema
│
└── packages/
    ├── shared/                     # Shared TypeScript types & parseJSON util
    ├── japanese-processing/        # Segmenter, romaji, furigana utilities
    ├── alignment/                  # Lyric timing types (sync scaffold)
    ├── providers/                  # Groq, Jotoba, Jisho API clients
    └── anki/                       # Anki card builder & deep-link URL generator
```

## Quick Start

### Prerequisites

- Node.js 20+
- pnpm 9+

### 1. Clone & install

```bash
git clone <repo-url>
cd japanese-song-lyrics-translator
pnpm install
```

### 2. Configure environment

```bash
cp .env.example apps/web/.env.local
# Edit apps/web/.env.local — set GROQ_API_KEY and DATABASE_URL
```

### 3. Set up the database

```bash
pnpm --filter @japanese-lyrics/web db:push
```

### 4. Run development server

```bash
pnpm dev
# Opens http://localhost:3000
```

## Environment Variables

Copy `.env.example` to `apps/web/.env.local`:

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | SQLite path, e.g. `file:./dev.db` |
| `GROQ_API_KEY` | Optional | Enables AI analysis (Llama 3.3 via Groq) |
| `YOUTUBE_API_KEY` | Optional | Fetches video title/artist metadata |
| `NEXT_PUBLIC_APP_URL` | Optional | Public URL for the app |

## Package Scripts

```bash
pnpm dev              # Start all packages in watch mode
pnpm build            # Production build
pnpm lint             # ESLint across all packages
pnpm type-check       # TypeScript check across all packages
pnpm format           # Prettier format

# Database (run from apps/web or use --filter)
pnpm --filter @japanese-lyrics/web db:push     # Sync schema → DB (dev)
pnpm --filter @japanese-lyrics/web db:migrate  # Create migration
pnpm --filter @japanese-lyrics/web db:studio   # Open Prisma Studio
```

## Docker

```bash
# Build and run
docker compose up --build

# With your API key
GROQ_API_KEY=gsk_xxx docker compose up
```

## Packages

### `@japanese-lyrics/shared`
Shared TypeScript types (`Token`, `AnalysisResult`, `LyricLine`, `Song`) and the `parseJSON` utility for robust AI response parsing.

### `@japanese-lyrics/japanese-processing`
Ported from the original `japanese-translator` project:
- `segmentText(text)` — Japanese word segmentation via `Intl.Segmenter`
- `toRomaji(kana)` — Hepburn romanization converter
- `furiganaToRuby(text)` — Converts `食(た)べる` format to `<ruby>` HTML
- `alignFurigana(surface, reading)` — Strips kana suffix for precise kanji ruby
- `buildRubyHTML(surface, reading, furigana)` — Full ruby HTML builder
- `buildRubyFromTokens(tokens)` — Sentence-level ruby from token array
- `looksLikeReading(s)` — Heuristic: detects kana-only strings

### `@japanese-lyrics/providers`
API clients ported from the original project:
- `analyzeWithGroq(text, apiKey)` — Full AI analysis via Groq / Llama 3.3
- `lookupJotoba(word)` — Jotoba dictionary lookup
- `lookupJisho(word)` — Jisho dictionary lookup (with CORS proxy fallback)
- `basicLookup(word)` — Tries Jotoba then Jisho

### `@japanese-lyrics/anki`
Anki integration ported from the original project:
- `buildAnkiFront(token)` — Styled HTML for card front
- `buildAnkiBack(token, ctx)` — Styled HTML for card back (with sentence context)
- `buildAnkiUrl(token, opts)` — `anki://x-callback-url/addnote` deep link

### `@japanese-lyrics/alignment`
Placeholder for future lyric timing synchronization:
- Whisper-based forced alignment (planned)
- Manual timestamp editor helpers (planned)
- Confidence scoring (planned)

## Roadmap

- [ ] YouTube metadata fetching (title, artist, thumbnail)
- [ ] Lyrics import (manual paste / auto-fetch)
- [ ] AI analysis pipeline for all lyric lines
- [ ] Lyric timing synchronization (Whisper forced alignment)
- [ ] Karaoke highlighting in sync with video playback
- [ ] Word-click analysis panel
- [ ] Anki export from player view

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript 5 |
| Styling | Tailwind CSS + shadcn/ui |
| Database | Prisma + SQLite |
| AI | Groq API (Llama 3.3 70B) |
| Monorepo | pnpm workspaces + Turborepo |
| Container | Docker + Compose |
