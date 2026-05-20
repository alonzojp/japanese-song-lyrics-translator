/**
 * CSV / TSV export for Anki's text file importer.
 *
 * Anki can import tab-separated or semicolon-separated text files.
 * The first line can specify #separator, #html, #tags columns, and #deck.
 *
 * Format:
 *   # Line 1: metadata directives
 *   front_field<TAB>back_field<TAB>tags
 *
 * Usage:
 *   File → Import → choose .txt file → set separator = Tab
 *
 * This is the most universally compatible format — works on all platforms
 * that have Anki desktop.
 */
import type { ExportCard, ExportResult } from "./types.js";

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Escape a field value for tab-separated format. */
function escapeTsv(value: string): string {
  // Replace literal tabs and newlines; Anki handles HTML otherwise
  return value.replace(/\t/g, "    ").replace(/\r?\n/g, "<br>");
}

/** Strip HTML tags for a plain-text preview (used for CSV without HTML). */
function stripHtml(html: string): string {
  return html.replace(/<[^>]+>/g, "").trim();
}

// ── Generators ─────────────────────────────────────────────────────────────────

/**
 * Generate Anki-compatible TSV content.
 *
 * The header comments tell Anki exactly how to interpret the file:
 *   #separator:tab
 *   #html:true          — fields contain HTML
 *   #notetype:Basic     — note type to use
 *   #deck:Deck Name     — target deck
 *   #tags column:3      — column 3 is the tags column
 */
export function cardsToAnkiTsv(
  cards:    ExportCard[],
  deckName  = "Japanese Lyrics",
): string {
  const lines: string[] = [
    "#separator:tab",
    "#html:true",
    `#notetype:Basic`,
    `#deck:${deckName}`,
    "#tags column:3",
    "",  // blank line before data
  ];

  for (const card of cards) {
    const front = escapeTsv(card.front);
    const back  = escapeTsv(card.back);
    const tags  = (card.tags ?? []).join(" ");
    lines.push(`${front}\t${back}\t${tags}`);
  }

  return lines.join("\n");
}

/**
 * Generate a simple CSV with stripped HTML — for spreadsheet applications
 * or manual inspection. Not for direct Anki import.
 */
export function cardsToPlainCsv(cards: ExportCard[]): string {
  const lines = ["Front,Back,Tags,Deck"];
  for (const card of cards) {
    const front = `"${stripHtml(card.front).replace(/"/g, '""')}"`;
    const back  = `"${stripHtml(card.back).replace(/"/g, '""')}"`;
    const tags  = `"${(card.tags ?? []).join(" ")}"`;
    const deck  = `"${card.deckName ?? ""}"`;
    lines.push(`${front},${back},${tags},${deck}`);
  }
  return lines.join("\n");
}

// ── Download helpers ───────────────────────────────────────────────────────────

/**
 * Create a Blob for the Anki TSV export.
 * The BOM (EF BB BF) helps some apps correctly detect UTF-8.
 */
export function createTsvBlob(content: string): Blob {
  const bom = "﻿";
  return new Blob([bom + content], { type: "text/plain;charset=utf-8" });
}

/**
 * Generate an Anki-importable TSV export result.
 * Call `result.blob` and trigger a browser download.
 */
export function exportToCsv(
  cards:    ExportCard[],
  deckName  = "Japanese Lyrics",
): ExportResult {
  if (!cards.length) {
    return { method: "csv", status: "error", message: "No cards to export." };
  }

  const content  = cardsToAnkiTsv(cards, deckName);
  const blob     = createTsvBlob(content);
  const filename = `${deckName.replace(/[^a-z0-9]/gi, "_")}_${Date.now()}.txt`;

  return {
    method:   "csv",
    status:   "success",
    count:    cards.length,
    blob,
    filename,
    message:  `${cards.length} card${cards.length !== 1 ? "s" : ""} ready. Import via Anki → File → Import.`,
  };
}

// ── Browser download ───────────────────────────────────────────────────────────

/** Trigger a browser file download from a Blob. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement("a");
  a.href    = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
