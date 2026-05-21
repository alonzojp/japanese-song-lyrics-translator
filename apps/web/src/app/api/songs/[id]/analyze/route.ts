import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { analyzeLine } from "@japanese-lyrics/japanese-processing";
import type { Token } from "@japanese-lyrics/shared";
import type { JapaneseToken } from "@japanese-lyrics/japanese-processing";

interface RouteParams {
  params: { id: string };
}

function toSharedToken(t: JapaneseToken): Token {
  return {
    surface:         t.surface,
    reading:         t.reading     || null,
    furigana:        t.furigana,
    romaji:          t.romaji      || null,
    part_of_speech:  t.partOfSpeech,
    dictionary_form: t.baseForm    || null,
    english:         t.meaning,
    jlpt:            t.jlpt,
    notes:           null,
  };
}

export async function POST(_req: Request, { params }: RouteParams) {
  const lines = await prisma.lyricLine.findMany({
    where:   { songId: params.id },
    orderBy: { lineIndex: "asc" },
  });

  if (lines.length === 0) {
    return NextResponse.json({ analyzed: 0 });
  }

  const results = await Promise.allSettled(
    lines.map((l) => analyzeLine(l.japanese, { musicMode: true }))
  );

  let analyzed = 0;
  for (let i = 0; i < lines.length; i++) {
    const r = results[i]!;
    if (r.status !== "fulfilled") continue;

    const { tokens: nlpTokens, furigana } = r.value;
    const tokens: Token[] = nlpTokens.map(toSharedToken);

    await prisma.lyricLine.update({
      where: { id: lines[i]!.id },
      data: {
        tokens:   JSON.stringify(tokens),
        analysis: JSON.stringify({ furigana }),
      },
    });
    analyzed++;
  }

  return NextResponse.json({ analyzed });
}
