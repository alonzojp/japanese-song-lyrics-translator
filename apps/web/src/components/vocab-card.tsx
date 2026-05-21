"use client";

import type { Token } from "@japanese-lyrics/shared";
import type { ExportCard } from "@japanese-lyrics/anki";
import { buildAnkiFront, buildAnkiBack } from "@japanese-lyrics/anki";

// ── JLPT colour map ────────────────────────────────────────────────────────────

const JLPT_STYLE: Record<string, { bg: string; text: string }> = {
  N5: { bg: "bg-green-100 dark:bg-green-900/30",   text: "text-green-700 dark:text-green-400" },
  N4: { bg: "bg-sky-100 dark:bg-sky-900/30",       text: "text-sky-700 dark:text-sky-400" },
  N3: { bg: "bg-indigo-100 dark:bg-indigo-900/30", text: "text-indigo-700 dark:text-indigo-400" },
  N2: { bg: "bg-purple-100 dark:bg-purple-900/30", text: "text-purple-700 dark:text-purple-400" },
  N1: { bg: "bg-red-100 dark:bg-red-900/30",       text: "text-red-700 dark:text-red-400" },
};

// ── Card builders ──────────────────────────────────────────────────────────────

function sanitizeTag(s: string): string {
  return s.replace(/[\s　]/g, "-").replace(/[^\w\-]/g, "").slice(0, 40).toLowerCase();
}

export function makeWordCard(
  token:     Token,
  lineText:  string,
  songTitle: string | null,
  startTime?: number,
): ExportCard {
  const enriched: Token = {
    ...token,
    notes: [
      songTitle ? `From: ${songTitle}` : null,
      startTime != null ? `@${Math.floor(startTime)}s` : null,
    ]
      .filter(Boolean)
      .join("  ·  ") || null,
  };

  const tags = [
    "japanese-lyrics",
    ...(token.jlpt  ? [token.jlpt.toLowerCase()]       : []),
    ...(songTitle   ? [sanitizeTag(songTitle)]          : []),
  ];

  return {
    front:    buildAnkiFront(enriched),
    back:     buildAnkiBack(enriched, { sentence: lineText }),
    tags,
  };
}

export function makeSentenceCards(
  lineTokens: Token[],
  lineText:   string,
  songTitle:  string | null,
  startTime?: number,
): ExportCard[] {
  return lineTokens
    .filter((t) => t.english?.trim())
    .map((t) => makeWordCard(t, lineText, songTitle, startTime));
}

// ── Props ──────────────────────────────────────────────────────────────────────

export interface VocabCardProps {
  token:       Token;
  tokenIndex:  number;
  lineTokens:  Token[];
  lineText:    string;
  startTime?:  number;
  songTitle:   string | null;
  onClose:     () => void;
  onExport:    (cards: ExportCard[]) => void;
}

// ── Component ──────────────────────────────────────────────────────────────────

export function VocabCard({
  token,
  tokenIndex,
  lineTokens,
  lineText,
  startTime,
  songTitle,
  onClose,
  onExport,
}: VocabCardProps) {
  const jlptStyle  = token.jlpt ? JLPT_STYLE[token.jlpt] : null;
  const displayWord = token.dictionary_form ?? token.surface;
  const sentenceCardCount = lineTokens.filter((t) => t.english?.trim()).length;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Card — slides up from bottom on mobile, centered on desktop */}
      <div
        className="fixed bottom-0 left-0 right-0 z-50 mx-auto max-w-md rounded-t-2xl bg-white px-5 pb-8 pt-5 shadow-2xl dark:bg-zinc-900 sm:bottom-auto sm:left-1/2 sm:top-1/2 sm:w-full sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Drag handle (mobile) */}
        <div className="mx-auto mb-4 h-1 w-10 rounded-full bg-zinc-300 dark:bg-zinc-600 sm:hidden" />

        {/* Header row */}
        <div className="mb-3 flex items-start justify-between gap-3">
          <div className="min-w-0">
            {/* Word with optional furigana */}
            <div className="text-4xl font-bold leading-snug tracking-tight">
              {token.furigana ? (
                <ruby className="ruby-pronunciation">
                  {token.surface}
                  <rt className="text-sm font-normal text-zinc-400 dark:text-zinc-500">
                    {token.reading}
                  </rt>
                </ruby>
              ) : (
                displayWord
              )}
            </div>
            {token.romaji && (
              <p className="mt-1 text-sm text-zinc-400">{token.romaji}</p>
            )}
          </div>

          <button
            onClick={onClose}
            className="mt-1 shrink-0 rounded-full p-1.5 text-zinc-400 transition hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* JLPT + POS badges */}
        <div className="mb-3 flex flex-wrap gap-2">
          {token.jlpt && jlptStyle && (
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${jlptStyle.bg} ${jlptStyle.text}`}
            >
              {token.jlpt}
            </span>
          )}
          {token.part_of_speech && (
            <span className="rounded-full border border-zinc-200 bg-zinc-50 px-2.5 py-0.5 text-xs text-zinc-500 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-400">
              {token.part_of_speech}
            </span>
          )}
        </div>

        {/* Meaning */}
        {token.english ? (
          <p className="mb-4 text-xl font-semibold text-zinc-800 dark:text-zinc-100">
            {token.english}
          </p>
        ) : (
          <p className="mb-4 text-sm italic text-zinc-400">No definition found</p>
        )}

        {/* Lyric context */}
        <div className="mb-5 rounded-xl border border-zinc-200 bg-zinc-50/80 px-4 py-3 dark:border-zinc-700 dark:bg-zinc-800/40">
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-zinc-400">
            Lyric context
            {startTime != null && (
              <span className="ml-2 normal-case tracking-normal">
                @{Math.floor(startTime / 60)}:{String(Math.floor(startTime % 60)).padStart(2, "0")}
              </span>
            )}
          </p>
          <p className="font-japanese text-base leading-loose">
            {lineTokens.map((t, i) => (
              <span
                key={i}
                className={
                  i === tokenIndex
                    ? "rounded bg-primary/15 px-0.5 font-semibold text-primary"
                    : ""
                }
              >
                {t.surface}
              </span>
            ))}
          </p>
        </div>

        {/* Export buttons */}
        <div className="flex gap-2.5">
          <button
            onClick={() =>
              onExport([makeWordCard(token, lineText, songTitle, startTime)])
            }
            className="flex-1 rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-primary/90 active:scale-95"
          >
            Export word
          </button>
          {sentenceCardCount > 0 && (
            <button
              onClick={() =>
                onExport(makeSentenceCards(lineTokens, lineText, songTitle, startTime))
              }
              className="flex-1 rounded-xl border border-zinc-300 px-4 py-2.5 text-sm font-medium transition hover:bg-zinc-50 active:scale-95 dark:border-zinc-600 dark:hover:bg-zinc-800"
            >
              Export line&nbsp;
              <span className="text-zinc-400">({sentenceCardCount})</span>
            </button>
          )}
        </div>
      </div>
    </>
  );
}
