# Japanese Song Lyrics Translator

Learn Japanese through music. Paste a YouTube URL to get furigana readings, word-by-word translations, and karaoke-style lyric highlighting — all running locally with no paid APIs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                        │
│  Next.js frontend (React + TypeScript + Tailwind)               │
│  • YouTube URL input                                            │
│  • Job progress polling (every 2.5 s)                           │
│  • Karaoke lyrics display                                       │
└──────────────────┬──────────────────────────────────────────────┘
                   │  HTTP
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Next.js API Gateway (lightweight)                              │
│  POST /api/jobs         → submit job to Python worker           │
│  GET  /api/jobs/[id]    → proxy poll + sync lyrics to DB        │
│  POST /api/songs        → upsert song record                    │
│  GET  /api/songs/[id]/lyrics → serve cached lyrics              │
│                                                                 │
│  SQLite (Prisma)  — songs + lyric lines                        │
└──────────────────┬──────────────────────────────────────────────┘
                   │  HTTP  (WORKER_URL)
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Python FastAPI Worker  (services/worker)                       │
│                                                                 │
│  Job queue (SQLite)                                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  queued → processing → completed / failed                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Pipeline steps (background thread)                            │
│  1. download   — yt-dlp   → audio.wav                          │
│  2. separate   — Demucs   → vocals.wav   (skippable)           │
│  3. transcribe — WhisperX → word-level timestamps              │
│  4. align      — post-process → lyrics.json                    │
│                                                                 │
│  Filesystem cache  /cache/{youtubeId}/                         │
│    audio.wav  |  vocals.wav  |  lyrics.json                    │
└─────────────────────────────────────────────────────────────────┘
```

### Design principles
- **Zero paid APIs by default** — local Whisper, local Demucs, local everything
- **Filesystem caching** — re-running a song skips already-completed steps
- **GPU optional** — falls back to CPU automatically (slower but works everywhere)
- **OpenAI API is entirely optional** — used only if you set `GROQ_API_KEY` for AI word analysis

---

## Monorepo Structure

```
japanese-song-lyrics-translator/
├── apps/
│   └── web/                        # Next.js 14 app
│       ├── src/app/                # Pages + API routes
│       ├── src/components/         # React components + shadcn/ui
│       ├── src/lib/                # Prisma client, worker client
│       └── prisma/schema.prisma    # SQLite schema (songs + lyric lines)
│
├── services/
│   └── worker/                     # Python FastAPI audio processing worker
│       ├── main.py                 # FastAPI app + routes
│       ├── pipeline.py             # Job orchestrator
│       ├── queue.py                # SQLite job queue
│       ├── models.py               # Pydantic models (mirrors TS types)
│       ├── config.py               # Env-var config
│       └── steps/
│           ├── download.py         # yt-dlp
│           ├── separate.py         # Demucs vocal isolation
│           ├── transcribe.py       # WhisperX ASR + alignment
│           └── align.py            # Post-processing → lyrics.json
│
└── packages/
    ├── shared/                     # TS types + API contract (Job, TranscribedLine…)
    ├── japanese-processing/        # Furigana, romaji, segmenter
    ├── providers/                  # Groq, Jotoba, Jisho clients
    ├── anki/                       # Anki card builder
    └── alignment/                  # Alignment type scaffolding
```

---

## Quick Start

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Node.js | 20+ | |
| pnpm | 9+ | `npm install -g pnpm` |
| Python | 3.10+ | For the worker |
| ffmpeg | any | `winget install ffmpeg` / `brew install ffmpeg` |

### 1. Clone & install Node deps

```bash
git clone https://github.com/alonzojp/japanese-song-lyrics-translator
cd japanese-song-lyrics-translator
pnpm install
```

### 2. Configure Next.js

```bash
cp apps/web/.env.local.example apps/web/.env.local
# Edit apps/web/.env.local — set GROQ_API_KEY if you want AI word analysis
```

### 3. Set up the database

```bash
pnpm --filter @japanese-lyrics/web db:push
```

### 4. Set up the Python worker

```bash
cd services/worker
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install PyTorch first (CPU build — faster to download):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then the rest:
pip install -r requirements.txt

# Copy and edit env:
cp .env.example .env
```

### 5. Run both services

**Terminal 1 — Next.js:**
```bash
pnpm dev
```

**Terminal 2 — Python worker:**
```bash
cd services/worker
python main.py
```

Open http://localhost:3000, paste a Japanese song URL, click **Start Processing**.

---

## Environment Variables

### `apps/web/.env.local`

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | `file:./dev.db` | SQLite path |
| `WORKER_URL` | Yes | `http://localhost:8000` | Python worker URL |
| `GROQ_API_KEY` | No | — | Enables AI word analysis (optional) |
| `NEXT_PUBLIC_APP_URL` | No | `http://localhost:3000` | Public URL |

### `services/worker/.env`

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Whisper model size (tiny/base/small/medium/large-v3) |
| `WHISPER_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `WHISPER_COMPUTE_TYPE` | `auto` | `int8` (CPU) \| `float16` (GPU) |
| `SKIP_VOCAL_SEPARATION` | `false` | Skip Demucs (faster, less accurate on noisy songs) |
| `CACHE_DIR` | `./cache` | Where to store audio + lyrics files |

---

## Job Queue States

```
POST /api/jobs
      ↓
   queued         # job created, waiting for thread
      ↓
  processing      # one of: download → separate → transcribe → align
      ↓
  completed  ──→  lyrics stored in SQLite, served to frontend
  failed     ──→  error message returned, retry available
```

---

## Docker

```bash
# Build and start both services:
docker compose up --build

# With custom Whisper model (medium = faster, less accurate):
WHISPER_MODEL=medium docker compose up

# With Groq AI analysis:
GROQ_API_KEY=gsk_xxx docker compose up
```

### GPU support

Uncomment the `worker-gpu` service in `docker-compose.yml` (requires NVIDIA Docker runtime).

---

## Package Scripts

```bash
pnpm dev                                          # start Next.js dev server
pnpm build                                        # production build
pnpm type-check                                   # TypeScript check all packages
pnpm lint                                         # ESLint all packages

pnpm --filter @japanese-lyrics/web db:push        # sync schema → SQLite (dev)
pnpm --filter @japanese-lyrics/web db:studio      # open Prisma Studio
```

---

## Roadmap

- [x] Monorepo scaffolding
- [x] YouTube URL input + song storage
- [x] Python worker architecture (yt-dlp + Demucs + WhisperX)
- [x] Job queue (queued → processing → completed/failed)
- [x] Live progress polling in the UI
- [ ] Lyrics display with word-level highlighting
- [ ] AI word analysis (furigana + translation per line)
- [ ] Anki export from player view
- [ ] Manual lyrics editor + timestamp correction
- [ ] YouTube metadata fetch (title, artist)
