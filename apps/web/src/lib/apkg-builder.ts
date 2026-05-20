/**
 * Server-side .apkg generator.
 *
 * .apkg format:
 *   ZIP archive containing:
 *     collection.anki2  — SQLite database (Anki2 schema)
 *     media             — JSON mapping of media filenames (empty for text cards)
 *
 * Uses sql.js (SQLite in WASM) for the database and fflate for ZIP.
 * Runs only in Node.js API routes — not in the browser bundle.
 */

import { zipSync } from "fflate";
import type { ExportCard } from "@japanese-lyrics/anki";

// ── Anki2 schema ───────────────────────────────────────────────────────────────

const CREATE_SCHEMA = `
CREATE TABLE col (
  id    integer primary key,
  crt   integer not null,
  mod   integer not null,
  scm   integer not null,
  ver   integer not null,
  dty   integer not null,
  usn   integer not null,
  ls    integer not null,
  conf  text    not null,
  models text   not null,
  decks text    not null,
  dconf text    not null,
  tags  text    not null
);
CREATE TABLE notes (
  id    integer primary key,
  guid  text    not null,
  mid   integer not null,
  mod   integer not null,
  usn   integer not null,
  tags  text    not null,
  flds  text    not null,
  sfld  integer not null,
  csum  integer not null,
  flags integer not null,
  data  text    not null
);
CREATE TABLE cards (
  id      integer primary key,
  nid     integer not null,
  did     integer not null,
  ord     integer not null,
  mod     integer not null,
  usn     integer not null,
  type    integer not null,
  queue   integer not null,
  due     integer not null,
  ivl     integer not null,
  factor  integer not null,
  reps    integer not null,
  lapses  integer not null,
  left    integer not null,
  odue    integer not null,
  odid    integer not null,
  flags   integer not null,
  data    text    not null
);
CREATE TABLE revlog (
  id      integer primary key,
  cid     integer not null,
  usn     integer not null,
  ease    integer not null,
  ivl     integer not null,
  lastIvl integer not null,
  factor  integer not null,
  time    integer not null,
  type    integer not null
);
CREATE TABLE graves (usn integer not null, oid integer not null, type integer not null);
CREATE INDEX ix_notes_usn  on notes (usn);
CREATE INDEX ix_cards_usn  on cards (usn);
CREATE INDEX ix_cards_nid  on cards (nid);
CREATE INDEX ix_cards_sched on cards (did, queue, due);
CREATE INDEX ix_revlog_usn on revlog (usn);
CREATE INDEX ix_revlog_cid on revlog (cid);
`;

// ── ID and checksum helpers ────────────────────────────────────────────────────

function nowSec(): number {
  return Math.floor(Date.now() / 1000);
}

function randomId(): number {
  return Date.now() * 1000 + Math.floor(Math.random() * 1000);
}

/** Generate a random guid for Anki notes (10 chars, base91 chars). */
function randomGuid(): string {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+,-./:;<=>?@[]^_`{|}~";
  let s = "";
  for (let i = 0; i < 10; i++) {
    s += chars[Math.floor(Math.random() * chars.length)];
  }
  return s;
}

/** Anki's note checksum: first 8 hex digits of the SHA-1 of the sort field. */
async function fieldChecksum(text: string): Promise<number> {
  const data    = new TextEncoder().encode(text);
  const hashBuf = await crypto.subtle.digest("SHA-1", data);
  const hex     = Array.from(new Uint8Array(hashBuf)).map((b) => b.toString(16).padStart(2, "0")).join("");
  return parseInt(hex.slice(0, 8), 16);
}

// ── Anki config payloads ───────────────────────────────────────────────────────

function buildModelJson(modelId: number, deckId: number): Record<string, unknown> {
  return {
    [String(modelId)]: {
      id:     String(modelId),
      name:   "Japanese Lyrics Basic",
      type:   0,
      mod:    nowSec(),
      usn:    -1,
      did:    deckId,
      flds: [
        { name: "Front", ord: 0, sticky: false, rtl: false, font: "Arial", size: 20 },
        { name: "Back",  ord: 1, sticky: false, rtl: false, font: "Arial", size: 20 },
      ],
      tmpls: [
        {
          name:  "Card 1",
          ord:   0,
          qfmt:  "{{Front}}",
          afmt:  "{{FrontSide}}<hr id='answer'>{{Back}}",
          bqfmt: "",
          bafmt: "",
          did:   null,
          bfont: "",
          bsize: 0,
        },
      ],
      css: `.card {
  font-family: "Hiragino Sans", "Noto Sans JP", Arial, sans-serif;
  font-size: 20px;
  text-align: center;
  color: black;
  background-color: white;
}
ruby rt { font-size: 0.5em; color: #666; }`,
      latexPre:  "\\documentclass[12pt]{article}\\usepackage{amssymb,amsmath}\\begin{document}",
      latexPost: "\\end{document}",
      sortf: 0,
      tags: [],
      req: [[0, "any", [0]]],
    },
  };
}

function buildDeckJson(deckId: number, deckName: string): Record<string, unknown> {
  return {
    [String(deckId)]: {
      id:        deckId,
      name:      deckName,
      desc:      "Exported from Japanese Song Lyrics Translator",
      dyn:       0,
      conf:      1,
      usn:       -1,
      collapsed: false,
      browserCollapsed: false,
      newToday:  [0, 0],
      revToday:  [0, 0],
      lrnToday:  [0, 0],
      timeToday: [0, 0],
      mod:       nowSec(),
    },
  };
}

const DECK_CONF = JSON.stringify({
  "1": {
    id: 1,
    name: "Default",
    replayq: true,
    lapse: { leechFails: 8, minInt: 1, delays: [10], leechAction: 0, mult: 0 },
    rev: { perDay: 200, fuzz: 0.05, ivlFct: 1, maxIvl: 36500, ease4: 1.3, bury: true, minSpace: 1 },
    timer: 0,
    maxTaken: 60,
    usn: 0,
    new: { perDay: 20, delays: [1, 10], separate: true, ints: [1, 4, 7], initialFactor: 2500, bury: true, order: 1 },
    mod: 0,
    autoplay: true,
  },
});

const COL_CONF = JSON.stringify({
  nextPos:    1,
  estTimes:   true,
  activeDecks: [1],
  sortType:   "noteFld",
  timeLim:    0,
  sortBackwards: false,
  addToCur:   true,
  dayLearnFirst: false,
  newBury:    true,
  newSpread:  0,
  dueCounts:  true,
  curModel:   null,
  collapseTime: 1200,
});

// ── Main builder ───────────────────────────────────────────────────────────────

export async function buildApkg(
  cards:    ExportCard[],
  deckName  = "Japanese Lyrics",
): Promise<Uint8Array> {
  const { default: initSqlJs } = await import("sql.js");
  // In Node.js, sql.js can find its WASM automatically
  const SQL = await initSqlJs();
  const db  = new SQL.Database();

  db.run(CREATE_SCHEMA);

  const now     = nowSec();
  const deckId  = randomId();
  const modelId = randomId();

  // Insert collection row
  db.run(
    `INSERT INTO col VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)`,
    [
      1, now, now, now, 11, 0, 0, 0,
      COL_CONF,
      JSON.stringify(buildModelJson(modelId, deckId)),
      JSON.stringify(buildDeckJson(deckId, deckName)),
      DECK_CONF,
      "{}",
    ],
  );

  // Insert notes and cards
  const insertNote = db.prepare(
    `INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?,?,?)`
  );
  const insertCard = db.prepare(
    `INSERT INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`
  );

  let cardDue = 0;
  for (const card of cards) {
    const noteId   = randomId();
    const guid     = card.guid ?? randomGuid();
    const flds     = `${card.front}\x1f${card.back}`;
    const sfld     = card.front.replace(/<[^>]+>/g, "").slice(0, 100);
    const csum     = await fieldChecksum(sfld);
    const tags     = (card.tags ?? []).join(" ");

    insertNote.run([noteId, guid, modelId, now, -1, tags, flds, sfld, csum, 0, ""]);

    const cardId = randomId();
    insertCard.run([cardId, noteId, deckId, 0, now, -1, 0, 0, cardDue++, 0, 0, 0, 0, 0, 0, 0, 0, ""]);
  }

  insertNote.free();
  insertCard.free();

  const dbBytes = db.export();   // Uint8Array
  db.close();

  // ZIP into .apkg
  const apkgBytes = zipSync({
    "collection.anki2": dbBytes,
    "media":            new TextEncoder().encode("{}"),
  });

  return apkgBytes;
}
