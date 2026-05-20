/**
 * AnkiConnect HTTP client.
 *
 * AnkiConnect is an Anki add-on that exposes a local REST API on port 8765.
 * It ONLY works on desktop Anki — not on iOS/Android/AnkiWeb.
 *
 * All functions in this file must be called from the BROWSER (client-side),
 * not from Next.js API routes, because the localhost target is the USER's
 * machine, not the server.
 *
 * AnkiConnect add-on ID: 2055492159
 * Documentation: https://foosoft.net/projects/anki-connect/
 */
import type {
  AnkiConnectNote,
  AnkiConnectStatus,
  ExportCard,
  ExportResult,
} from "./types.js";

const ANKICONNECT_URL    = "http://localhost:8765";
const ANKICONNECT_VER    = 6;
const CONNECT_TIMEOUT_MS = 3000;

// ── Low-level invoke ───────────────────────────────────────────────────────────

interface AnkiConnectResponse<T> {
  result: T;
  error:  string | null;
}

async function invoke<T = unknown>(
  action: string,
  params: Record<string, unknown> = {},
): Promise<T> {
  const controller = new AbortController();
  const timer      = setTimeout(() => controller.abort(), CONNECT_TIMEOUT_MS);

  try {
    const res = await fetch(ANKICONNECT_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ action, version: ANKICONNECT_VER, params }),
      signal:  controller.signal,
    });
    const data = (await res.json()) as AnkiConnectResponse<T>;
    if (data.error) throw new Error(data.error);
    return data.result;
  } finally {
    clearTimeout(timer);
  }
}

// ── Detection ──────────────────────────────────────────────────────────────────

/**
 * Check whether AnkiConnect is available on this machine.
 * Must be called client-side.
 */
export async function detectAnkiConnect(): Promise<AnkiConnectStatus> {
  try {
    const version = await invoke<number>("version");
    return { available: true, version };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "unreachable";
    const isAbort = msg === "AbortError" || msg.includes("abort");
    return {
      available: false,
      error: isAbort
        ? "AnkiConnect timed out — is Anki open?"
        : "AnkiConnect not reachable — open Anki and install the AnkiConnect add-on",
    };
  }
}

// ── Deck management ────────────────────────────────────────────────────────────

export async function ensureDeck(name: string): Promise<void> {
  await invoke("createDeck", { deck: name });
}

export async function getDeckNames(): Promise<string[]> {
  return invoke<string[]>("deckNames");
}

// ── Note operations ────────────────────────────────────────────────────────────

function cardToNote(card: ExportCard, defaultDeck: string): AnkiConnectNote {
  return {
    deckName:  card.deckName ?? defaultDeck,
    modelName: card.noteType ?? "Basic",
    fields: {
      Front: card.front,
      Back:  card.back,
    },
    options: { allowDuplicate: false },
    tags:    card.tags ?? [],
  };
}

/**
 * Add a single card via AnkiConnect.
 * Returns the new note ID, or null if duplicate.
 */
export async function addNote(
  card:        ExportCard,
  defaultDeck  = "Mining",
): Promise<ExportResult> {
  try {
    await ensureDeck(card.deckName ?? defaultDeck);
    const noteId = await invoke<number | null>("addNote", {
      note: cardToNote(card, defaultDeck),
    });
    return {
      method:   "ankiconnect",
      status:   "success",
      count:    noteId ? 1 : 0,
      noteIds:  noteId ? [noteId] : [],
      message:  noteId ? "Card added to Anki." : "Duplicate — card already exists.",
    };
  } catch (err) {
    return {
      method:  "ankiconnect",
      status:  "error",
      message: `AnkiConnect error: ${err instanceof Error ? err.message : "unknown"}`,
    };
  }
}

/**
 * Add multiple cards in a single batch request.
 */
export async function addNotes(
  cards:       ExportCard[],
  defaultDeck  = "Mining",
): Promise<ExportResult> {
  if (!cards.length) return { method: "ankiconnect", status: "success", count: 0 };

  try {
    const deckName = cards[0]?.deckName ?? defaultDeck;
    await ensureDeck(deckName);

    const notes   = cards.map((c) => cardToNote(c, defaultDeck));
    const results = await invoke<Array<number | null>>("addNotes", { notes });
    const added   = (results ?? []).filter(Boolean) as number[];

    return {
      method:  "ankiconnect",
      status:  "success",
      count:   added.length,
      noteIds: added,
      message: `${added.length} of ${cards.length} card${cards.length !== 1 ? "s" : ""} added.`,
    };
  } catch (err) {
    return {
      method:  "ankiconnect",
      status:  "error",
      message: `AnkiConnect error: ${err instanceof Error ? err.message : "unknown"}`,
    };
  }
}
