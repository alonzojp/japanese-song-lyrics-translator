/**
 * Shared types for the Anki export system.
 *
 * Designed to support all export targets:
 *   Desktop: AnkiConnect (localhost:8765), .apkg file
 *   All:     .csv/.tsv file (import manually in any Anki)
 *   Mobile:  .apkg download → Files app → Anki iOS/Android import
 *   AnkiWeb: guided workflow
 */

// ── Export methods ─────────────────────────────────────────────────────────────

export type ExportMethod =
  | "ankiconnect"   // Direct API to desktop Anki (requires AnkiConnect add-on)
  | "apkg"          // Download .apkg file (all platforms, import manually)
  | "csv"           // Download .csv file (Anki's text file importer)
  | "json"          // Download structured JSON (dev/custom import scripts)
  | "ankiweb";      // Guided AnkiWeb workflow

export type ExportStatus = "success" | "error" | "unavailable" | "pending";

// ── Card data ──────────────────────────────────────────────────────────────────

export interface ExportCard {
  /** HTML content for the front of the card */
  front:     string;
  /** HTML content for the back of the card */
  back:      string;
  tags?:     string[];
  deckName?: string;
  /** Unique identifier — used to avoid re-adding duplicate cards */
  guid?:     string;
  noteType?: string;
}

// ── Results ────────────────────────────────────────────────────────────────────

export interface ExportResult {
  method:      ExportMethod;
  status:      ExportStatus;
  message?:    string;
  /** Number of cards successfully exported */
  count?:      number;
  /** For file-based exports — the Blob to download */
  blob?:       Blob;
  /** Suggested filename for the download */
  filename?:   string;
  /** For AnkiConnect — the IDs of the added notes */
  noteIds?:    number[];
}

// ── AnkiConnect ────────────────────────────────────────────────────────────────

export interface AnkiConnectStatus {
  available: boolean;
  version?:  number;
  error?:    string;
}

export interface AnkiConnectNote {
  deckName:  string;
  modelName: string;
  fields:    Record<string, string>;
  options:   { allowDuplicate: boolean };
  tags:      string[];
}

// ── Platform detection ─────────────────────────────────────────────────────────

export type Platform = "desktop" | "ios" | "android" | "unknown";

export function detectPlatform(): Platform {
  if (typeof navigator === "undefined") return "unknown";
  const ua = navigator.userAgent.toLowerCase();
  if (/iphone|ipad|ipod/.test(ua))        return "ios";
  if (/android/.test(ua))                 return "android";
  if (/macintosh|windows|linux/.test(ua)) return "desktop";
  return "unknown";
}

/** True when AnkiConnect might work (desktop platforms). */
export function canUseAnkiConnect(): boolean {
  const p = detectPlatform();
  return p === "desktop" || p === "unknown";
}

// ── Method metadata ────────────────────────────────────────────────────────────

export interface ExportMethodInfo {
  method:      ExportMethod;
  label:       string;
  description: string;
  /** Best platforms for this method */
  platforms:   Platform[];
  requiresApp: boolean;
}

export const EXPORT_METHOD_INFO: ExportMethodInfo[] = [
  {
    method:      "ankiconnect",
    label:       "AnkiConnect (instant)",
    description: "Sends cards directly to your open desktop Anki. Requires the AnkiConnect add-on.",
    platforms:   ["desktop"],
    requiresApp: true,
  },
  {
    method:      "apkg",
    label:       "Download .apkg deck",
    description: "Download a deck file you can import into any Anki (desktop, iOS, Android).",
    platforms:   ["desktop", "ios", "android", "unknown"],
    requiresApp: false,
  },
  {
    method:      "csv",
    label:       "Download .txt (CSV)",
    description: "Plain text export. Import via File → Import in Anki desktop.",
    platforms:   ["desktop"],
    requiresApp: false,
  },
  {
    method:      "json",
    label:       "Download JSON",
    description: "Structured JSON export for custom import scripts or developer use.",
    platforms:   ["desktop", "ios", "android", "unknown"],
    requiresApp: false,
  },
  {
    method:      "ankiweb",
    label:       "AnkiWeb workflow",
    description: "Download a deck, import on desktop Anki, then sync to AnkiWeb and your phone.",
    platforms:   ["ios", "android", "unknown"],
    requiresApp: false,
  },
];
