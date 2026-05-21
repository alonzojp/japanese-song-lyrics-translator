/**
 * AnkiExporter — high-level export orchestrator with automatic fallback.
 *
 * Fallback chain (desktop):
 *   1. Try AnkiConnect (instant, no file needed)
 *   2. If unavailable → offer .apkg download
 *   3. If .apkg unavailable → offer CSV download
 *
 * Fallback chain (iOS/Android):
 *   1. .apkg download → open in Anki
 *   2. AnkiWeb sync guide
 *
 * All methods are non-throwing — errors are returned in ExportResult.
 */
import {
  detectAnkiConnect,
  addNote,
  addNotes,
} from "./connect.js";
import {
  exportToCsv,
  downloadBlob,
} from "./csv.js";
import { exportToJson } from "./json.js";
import {
  type ExportCard,
  type ExportMethod,
  type ExportResult,
  type AnkiConnectStatus,
  type Platform,
  canUseAnkiConnect,
  detectPlatform,
} from "./types.js";

// ── Options ────────────────────────────────────────────────────────────────────

export interface ExporterOptions {
  defaultDeck?:     string;
  /** Called to fetch a .apkg Blob from the server API */
  apkgFetcher?:     (cards: ExportCard[]) => Promise<Blob>;
  onProgress?:      (message: string) => void;
}

// ── AnkiExporter class ─────────────────────────────────────────────────────────

export class AnkiExporter {
  private readonly opts: Required<Pick<ExporterOptions, "defaultDeck">> & ExporterOptions;

  constructor(options: ExporterOptions = {}) {
    this.opts = {
      defaultDeck: options.defaultDeck ?? "Mining",
      ...options,
    };
  }

  private log(msg: string): void {
    this.opts.onProgress?.(msg);
  }

  // ── Detection ──────────────────────────────────────────────────────────────

  async detectAnkiConnect(): Promise<AnkiConnectStatus> {
    return detectAnkiConnect();
  }

  getPlatform(): Platform {
    return detectPlatform();
  }

  // ── Single method ──────────────────────────────────────────────────────────

  async export(
    cards:  ExportCard[],
    method: ExportMethod,
  ): Promise<ExportResult> {
    switch (method) {
      case "ankiconnect":
        return this._exportAnkiConnect(cards);
      case "apkg":
        return this._exportApkg(cards);
      case "csv":
        return this._exportCsv(cards);
      case "json":
        return this._exportJson(cards);
      case "ankiweb":
        return this._exportAnkiWeb(cards);
    }
  }

  // ── Smart fallback ─────────────────────────────────────────────────────────

  /**
   * Try AnkiConnect first; fall back to .apkg; fall back to CSV.
   * Returns the first successful result.
   */
  async exportWithFallback(cards: ExportCard[]): Promise<ExportResult[]> {
    const results: ExportResult[] = [];
    const platform = detectPlatform();

    if (canUseAnkiConnect()) {
      this.log("Checking for AnkiConnect…");
      const status = await detectAnkiConnect();
      if (status.available) {
        this.log("AnkiConnect found — sending cards…");
        const r = await this._exportAnkiConnect(cards);
        results.push(r);
        if (r.status === "success") return results;
      } else {
        results.push({
          method:  "ankiconnect",
          status:  "unavailable",
          message: status.error,
        });
      }
    }

    // .apkg is best for mobile and as desktop fallback
    if (this.opts.apkgFetcher) {
      this.log("Generating .apkg file…");
      const r = await this._exportApkg(cards);
      results.push(r);
      if (r.status === "success") return results;
    }

    // CSV as last resort (desktop only)
    if (platform !== "ios" && platform !== "android") {
      this.log("Generating CSV file…");
      results.push(this._exportCsv(cards));
    }

    return results;
  }

  // ── Private implementations ────────────────────────────────────────────────

  private async _exportAnkiConnect(cards: ExportCard[]): Promise<ExportResult> {
    if (cards.length === 1) {
      return addNote(cards[0]!, this.opts.defaultDeck);
    }
    return addNotes(cards, this.opts.defaultDeck);
  }

  private async _exportApkg(cards: ExportCard[]): Promise<ExportResult> {
    if (!this.opts.apkgFetcher) {
      return {
        method:  "apkg",
        status:  "unavailable",
        message: ".apkg export requires server-side support.",
      };
    }
    try {
      const blob     = await this.opts.apkgFetcher(cards);
      const deckName = cards[0]?.deckName ?? this.opts.defaultDeck;
      const filename = `${deckName.replace(/[^a-z0-9]/gi, "_")}_${Date.now()}.apkg`;
      downloadBlob(blob, filename);
      return {
        method:   "apkg",
        status:   "success",
        count:    cards.length,
        blob,
        filename,
        message:  `${cards.length} card${cards.length !== 1 ? "s" : ""} packaged. Open the downloaded file in Anki.`,
      };
    } catch (err) {
      return {
        method:  "apkg",
        status:  "error",
        message: `Failed to generate .apkg: ${err instanceof Error ? err.message : "unknown"}`,
      };
    }
  }

  private _exportCsv(cards: ExportCard[]): ExportResult {
    const result = exportToCsv(cards, this.opts.defaultDeck);
    if (result.status === "success" && result.blob && result.filename) {
      downloadBlob(result.blob, result.filename);
    }
    return result;
  }

  private _exportJson(cards: ExportCard[]): ExportResult {
    const result = exportToJson(cards, this.opts.defaultDeck);
    if (result.status === "success" && result.blob && result.filename) {
      downloadBlob(result.blob, result.filename);
    }
    return result;
  }

  private async _exportAnkiWeb(cards: ExportCard[]): Promise<ExportResult> {
    // AnkiWeb workflow = download .apkg → import to desktop → sync to AnkiWeb
    return this._exportApkg(cards);
  }
}

// ── Singleton factory ──────────────────────────────────────────────────────────

export function createExporter(options: ExporterOptions = {}): AnkiExporter {
  return new AnkiExporter(options);
}
