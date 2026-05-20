"use client";

import { useState, useEffect, useCallback } from "react";
import {
  detectPlatform,
  detectAnkiConnect,
  EXPORT_METHOD_INFO,
  type ExportCard,
  type ExportMethod,
  type Platform,
} from "@japanese-lyrics/anki";
import { downloadBlob, AnkiExporter } from "@japanese-lyrics/anki";

// ── Icon map (local, no extra dep) ─────────────────────────────────────────────

const METHOD_ICONS: Record<ExportMethod, string> = {
  ankiconnect: "⚡",
  apkg:        "📦",
  csv:         "📄",
  ankiweb:     "🌐",
};

// ── Props ──────────────────────────────────────────────────────────────────────

interface AnkiExportDialogProps {
  cards: ExportCard[];
  deckName?: string;
  open: boolean;
  onClose: () => void;
}

type DialogStep = "pick" | "ankiconnect-instructions" | "ankiweb-instructions" | "exporting" | "done" | "error";

// ── Component ──────────────────────────────────────────────────────────────────

export function AnkiExportDialog({ cards, deckName = "Japanese Lyrics", open, onClose }: AnkiExportDialogProps) {
  const [platform, setPlatform]             = useState<Platform>("unknown");
  const [ankiConnectOk, setAnkiConnectOk]   = useState<boolean | null>(null);
  const [step, setStep]                     = useState<DialogStep>("pick");
  const [errorMsg, setErrorMsg]             = useState("");
  const [successMsg, setSuccessMsg]         = useState("");

  useEffect(() => {
    if (!open) return;
    setStep("pick");
    setErrorMsg("");
    setSuccessMsg("");
    setPlatform(detectPlatform());
    detectAnkiConnect().then((s) => setAnkiConnectOk(s.available));
  }, [open]);

  const runAnkiConnect = useCallback(async () => {
    setStep("exporting");
    try {
      const exporter = new AnkiExporter({ defaultDeck: deckName });
      const result   = await exporter.export(cards, "ankiconnect");
      if (result.status === "success") {
        setSuccessMsg(`Added ${result.count ?? cards.length} card${(result.count ?? cards.length) !== 1 ? "s" : ""} to Anki.`);
        setStep("done");
      } else {
        setErrorMsg(result.message ?? "AnkiConnect export failed.");
        setStep("error");
      }
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : "Unknown error");
      setStep("error");
    }
  }, [cards, deckName]);

  const runDownload = useCallback(async (format: "apkg" | "csv") => {
    setStep("exporting");
    try {
      const res = await fetch("/api/anki/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cards, deckName, format }),
      });

      if (!res.ok) {
        const { error } = await res.json().catch(() => ({ error: "Server error" }));
        throw new Error(error ?? "Server error");
      }

      const blob     = await res.blob();
      const ext      = format === "apkg" ? "apkg" : "txt";
      const filename = `${deckName.replace(/\s+/g, "_")}.${ext}`;
      downloadBlob(blob, filename);

      setSuccessMsg(
        format === "apkg"
          ? "Deck downloaded. Open the .apkg file to import into Anki."
          : "File downloaded. In Anki: File → Import → select the .txt file.",
      );
      setStep("done");
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : "Unknown error");
      setStep("error");
    }
  }, [cards, deckName]);

  const handleMethodSelect = useCallback(async (method: ExportMethod) => {
    if (method === "ankiconnect") {
      if (!ankiConnectOk) {
        setStep("ankiconnect-instructions");
        return;
      }
      await runAnkiConnect();
      return;
    }
    if (method === "ankiweb") {
      setStep("ankiweb-instructions");
      return;
    }
    await runDownload(method as "apkg" | "csv");
  }, [ankiConnectOk, runAnkiConnect, runDownload]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="relative w-full max-w-md rounded-xl bg-white p-6 shadow-2xl dark:bg-zinc-900"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200"
          aria-label="Close"
        >
          ✕
        </button>

        <h2 className="mb-1 text-lg font-semibold">Export to Anki</h2>
        <p className="mb-4 text-sm text-zinc-500">
          {cards.length} card{cards.length !== 1 ? "s" : ""} · {deckName}
        </p>

        {step === "pick" && (
          <MethodPicker
            platform={platform}
            ankiConnectOk={ankiConnectOk}
            onSelect={handleMethodSelect}
          />
        )}

        {step === "ankiconnect-instructions" && (
          <AnkiConnectInstructions
            onRetry={() => {
              detectAnkiConnect().then((s) => {
                setAnkiConnectOk(s.available);
                if (s.available) runAnkiConnect();
              });
            }}
            onFallback={() => runDownload("apkg")}
          />
        )}

        {step === "ankiweb-instructions" && (
          <AnkiWebInstructions onDownloadCsv={() => runDownload("csv")} />
        )}

        {step === "exporting" && (
          <div className="flex flex-col items-center gap-3 py-6">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-blue-500" />
            <p className="text-sm text-zinc-500">Exporting…</p>
          </div>
        )}

        {step === "done" && (
          <div className="flex flex-col items-center gap-3 py-4 text-center">
            <div className="text-3xl">✅</div>
            <p className="text-sm text-zinc-700 dark:text-zinc-300">{successMsg}</p>
            <button
              onClick={onClose}
              className="mt-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
            >
              Done
            </button>
          </div>
        )}

        {step === "error" && (
          <div className="flex flex-col items-center gap-3 py-4 text-center">
            <div className="text-3xl">❌</div>
            <p className="text-sm text-red-600 dark:text-red-400">{errorMsg}</p>
            <div className="flex gap-2">
              <button
                onClick={() => setStep("pick")}
                className="rounded-lg border px-4 py-2 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-800"
              >
                Try another method
              </button>
              <button
                onClick={onClose}
                className="rounded-lg px-4 py-2 text-sm text-zinc-500 hover:text-zinc-700"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function MethodPicker({
  platform,
  ankiConnectOk,
  onSelect,
}: {
  platform: Platform;
  ankiConnectOk: boolean | null;
  onSelect: (m: ExportMethod) => void;
}) {
  const isMobile = platform === "ios" || platform === "android";

  return (
    <div className="flex flex-col gap-2">
      {isMobile && (
        <div className="mb-2 rounded-lg bg-amber-50 p-3 text-xs text-amber-800 dark:bg-amber-900/30 dark:text-amber-300">
          AnkiConnect is not available on {platform === "ios" ? "iPhone/iPad" : "Android"}.
          Use <strong>Download .apkg</strong> and open it with AnkiMobile / AnkiDroid.
        </div>
      )}

      {EXPORT_METHOD_INFO.map((info) => {
        const isAc = info.method === "ankiconnect";
        const badge =
          isAc && ankiConnectOk === true  ? "Connected" :
          isAc && ankiConnectOk === false ? "Not detected" :
          isAc                            ? "Checking…" : undefined;

        return (
          <button
            key={info.method}
            onClick={() => onSelect(info.method)}
            className="flex items-start gap-3 rounded-lg border border-zinc-200 p-3 text-left transition hover:border-blue-400 hover:bg-blue-50 dark:border-zinc-700 dark:hover:border-blue-500 dark:hover:bg-blue-950"
          >
            <span className="mt-0.5 text-xl">{METHOD_ICONS[info.method]}</span>
            <div className="flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium">{info.label}</span>
                {badge && (
                  <span
                    className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                      badge === "Connected"
                        ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400"
                        : badge === "Not detected"
                        ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400"
                        : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800"
                    }`}
                  >
                    {badge}
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">{info.description}</p>
            </div>
            <span className="mt-1 text-zinc-400">›</span>
          </button>
        );
      })}
    </div>
  );
}

function AnkiConnectInstructions({
  onRetry,
  onFallback,
}: {
  onRetry: () => void;
  onFallback: () => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm font-medium">Anki not detected on localhost:8765</p>
      <ol className="list-decimal space-y-1.5 pl-5 text-sm text-zinc-600 dark:text-zinc-400">
        <li>Open the Anki desktop app</li>
        <li>
          Install the <strong>AnkiConnect</strong> add-on (code{" "}
          <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">2055492159</code>)
        </li>
        <li>Restart Anki</li>
        <li>Click <strong>Retry</strong> below</li>
      </ol>
      <div className="flex gap-2">
        <button
          onClick={onRetry}
          className="flex-1 rounded-lg bg-blue-600 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          Retry
        </button>
        <button
          onClick={onFallback}
          className="flex-1 rounded-lg border py-2 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-800"
        >
          Download .apkg instead
        </button>
      </div>
    </div>
  );
}

function AnkiWebInstructions({ onDownloadCsv }: { onDownloadCsv: () => void }) {
  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm font-medium">Import into AnkiWeb</p>
      <ol className="list-decimal space-y-1.5 pl-5 text-sm text-zinc-600 dark:text-zinc-400">
        <li>Click <strong>Download CSV</strong> below</li>
        <li>
          Go to{" "}
          <a
            href="https://ankiweb.net"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline"
          >
            ankiweb.net
          </a>{" "}
          and sign in
        </li>
        <li>Click <strong>Import File</strong> and select the downloaded .txt file</li>
        <li>Set separator to <em>Tab</em> and enable <em>Allow HTML in fields</em></li>
        <li>Click <strong>Import</strong></li>
      </ol>
      <button
        onClick={onDownloadCsv}
        className="w-full rounded-lg bg-blue-600 py-2 text-sm font-medium text-white hover:bg-blue-700"
      >
        Download CSV
      </button>
    </div>
  );
}
