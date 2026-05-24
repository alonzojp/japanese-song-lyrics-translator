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
    id:           `line-${line.lineIndex}`,
    text:         line.japanese,
    startTime:    line.startTime ?? 0,
    endTime:      line.endTime   ?? 0,
    acousticStart: line.acousticStart ?? undefined,
    acousticEnd:   line.acousticEnd   ?? undefined,
    words:        line.words    ? (JSON.parse(line.words)    as WordTiming[]) : undefined,
    tokens:       line.tokens   ? (JSON.parse(line.tokens)   as Token[])     : undefined,
    furigana:     line.analysis ? (JSON.parse(line.analysis) as { furigana?: string }).furigana : undefined,
  }));

  return NextResponse.json({ lines: lyricLines, alignmentMeta });
}

export async function POST(req: Request, { params }: RouteParams) {
  const { text } = (await req.json()) as { text: string };
  if (!text?.trim()) return NextResponse.json({ error: "No lyrics text provided" }, { status: 400 });

  const song = await prisma.song.findUnique({
    where:  { id: params.id },
    select: { youtubeId: true },
  });
  if (!song) return NextResponse.json({ error: "Song not found" }, { status: 404 });

  await workerClient.uploadManualLyrics(song.youtubeId, text.trim());
  const result = await workerClient.matchLyrics(song.youtubeId);

  const lines = result.lines ?? [];
  if (lines.length > 0) {
    await prisma.lyricLine.deleteMany({ where: { songId: params.id } });
    await prisma.lyricLine.createMany({
      data: lines.map((line, i) => ({
        songId:       params.id,
        lineIndex:    i,
        startTime:    line.startTime,
        endTime:      line.endTime,
        acousticStart: (line as { acousticStart?: number }).acousticStart ?? null,
        acousticEnd:   (line as { acousticEnd?: number }).acousticEnd   ?? null,
        japanese:     line.text,
        words:        line.words ? JSON.stringify(line.words) : null,
      })),
    });
  }

  const lyricLines: LyricLine[] = lines.map((line, i) => ({
    id:        `line-${i}`,
    text:      line.text,
    startTime: line.startTime,
    endTime:   line.endTime,
    words:     line.words as WordTiming[] | undefined,
  }));

  return NextResponse.json({ lines: lyricLines });
}
