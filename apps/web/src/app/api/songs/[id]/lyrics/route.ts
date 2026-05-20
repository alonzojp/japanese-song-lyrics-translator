import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import type { LyricLine } from "@japanese-lyrics/shared";

interface RouteParams {
  params: { id: string };
}

export async function GET(_req: Request, { params }: RouteParams) {
  const lines = await prisma.lyricLine.findMany({
    where: { songId: params.id },
    orderBy: { lineIndex: "asc" },
  });

  const lyrics: LyricLine[] = lines.map((line) => ({
    index: line.lineIndex,
    startTime: line.startTime,
    endTime: line.endTime,
    japanese: line.japanese,
    tokens: line.tokens ? JSON.parse(line.tokens) : null,
    analysis: line.analysis ? JSON.parse(line.analysis) : null,
  }));

  return NextResponse.json(lyrics);
}
