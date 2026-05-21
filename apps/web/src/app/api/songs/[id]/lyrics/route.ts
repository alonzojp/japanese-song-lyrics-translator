import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { workerClient } from "@/lib/worker-client";
import type { LyricLine, WordTiming, AlignmentMeta, Token } from "@japanese-lyrics/shared";

interface RouteParams {
  params: { id: string };
}

export async function GET(_req: Request, { params }: RouteParams) {
  const lines = await prisma.lyricLine.findMany({
    where:   { songId: params.id },
    orderBy: { lineIndex: "asc" },
  });

  if (lines.length === 0) {
    return NextResponse.json({ lines: [], alignmentMeta: null });
  }

  // Try to fetch alignment metadata from the worker cache
  const song = await prisma.song.findUnique({
    where:  { id: params.id },
    select: { youtubeId: true },
  });

  let alignmentMeta: AlignmentMeta | null = null;
  if (song?.youtubeId) {
    try {
      const cached = await workerClient.getLyrics(song.youtubeId);
      // alignmentMeta may be embedded in cached lyrics result metadata
      alignmentMeta = (cached?.metadata?.alignmentMeta as AlignmentMeta) ?? null;
    } catch {
      // Worker offline or not cached — fine, just return DB lines
    }
  }

  const lyricLines: LyricLine[] = lines.map((line) => ({
    id:         `line-${line.lineIndex}`,
    text:       line.japanese,
    startTime:  line.startTime ?? 0,
    endTime:    line.endTime   ?? 0,
    words:      line.words    ? (JSON.parse(line.words)    as WordTiming[]) : undefined,
    tokens:     line.tokens   ? (JSON.parse(line.tokens)   as Token[])     : undefined,
    furigana:   line.analysis ? (JSON.parse(line.analysis) as { furigana?: string }).furigana : undefined,
  }));

  return NextResponse.json({ lines: lyricLines, alignmentMeta });
}
