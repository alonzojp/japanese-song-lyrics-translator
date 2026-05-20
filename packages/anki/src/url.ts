import type { Token } from "@japanese-lyrics/shared";
import { buildAnkiFront, buildAnkiBack, type AnkiCardContext } from "./card.js";

export interface AnkiUrlOptions extends AnkiCardContext {
  noteType?: string;
}

export function buildAnkiUrl(token: Token, opts: AnkiUrlOptions = {}): string {
  const profile = opts.profile ?? "User 1";
  const deck = opts.deck ?? "Mining";
  const noteType = opts.noteType ?? "Basic";

  const front = buildAnkiFront(token);
  const back = buildAnkiBack(token, opts);

  const enc = encodeURIComponent;
  return (
    "anki://x-callback-url/addnote" +
    `?profile=${enc(profile)}` +
    `&type=${enc(noteType)}` +
    `&deck=${enc(deck)}` +
    `&fldFront=${enc(front)}` +
    `&fldBack=${enc(back)}`
  );
}
