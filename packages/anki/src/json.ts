import type { ExportCard, ExportResult } from "./types.js";

export function exportToJson(
  cards:    ExportCard[],
  deckName  = "Japanese Lyrics",
): ExportResult {
  if (!cards.length) {
    return { method: "json", status: "error", message: "No cards to export." };
  }

  const payload = {
    deckName,
    exportedAt: new Date().toISOString(),
    count:      cards.length,
    cards:      cards.map((c) => ({
      front:    c.front,
      back:     c.back,
      tags:     c.tags ?? [],
      guid:     c.guid ?? undefined,
      deckName: c.deckName ?? deckName,
    })),
  };

  const json     = JSON.stringify(payload, null, 2);
  const blob     = new Blob([json], { type: "application/json;charset=utf-8" });
  const filename = `${deckName.replace(/[^a-z0-9]/gi, "_")}_${Date.now()}.json`;

  return {
    method:   "json",
    status:   "success",
    count:    cards.length,
    blob,
    filename,
    message:  `${cards.length} card${cards.length !== 1 ? "s" : ""} exported as JSON.`,
  };
}
