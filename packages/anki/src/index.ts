// ── Card building (existing) ───────────────────────────────────────────────────
export { buildAnkiFront, buildAnkiBack } from "./card.js";
export type { AnkiCardContext } from "./card.js";

// ── Deep link (existing, desktop-only) ────────────────────────────────────────
export { buildAnkiUrl } from "./url.js";
export type { AnkiUrlOptions } from "./url.js";

// ── Types ──────────────────────────────────────────────────────────────────────
export type {
  ExportCard,
  ExportMethod,
  ExportResult,
  ExportStatus,
  AnkiConnectStatus,
  AnkiConnectNote,
  Platform,
  ExportMethodInfo,
} from "./types.js";
export {
  EXPORT_METHOD_INFO,
  detectPlatform,
  canUseAnkiConnect,
} from "./types.js";

// ── AnkiConnect client ────────────────────────────────────────────────────────
export {
  detectAnkiConnect,
  addNote,
  addNotes,
  ensureDeck,
  getDeckNames,
} from "./connect.js";

// ── CSV export ────────────────────────────────────────────────────────────────
export {
  cardsToAnkiTsv,
  cardsToPlainCsv,
  createTsvBlob,
  exportToCsv,
  downloadBlob,
} from "./csv.js";

// ── Main exporter ─────────────────────────────────────────────────────────────
export { AnkiExporter, createExporter } from "./exporter.js";
export type { ExporterOptions } from "./exporter.js";
