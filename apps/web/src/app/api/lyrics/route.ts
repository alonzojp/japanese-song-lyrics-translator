import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { workerClient } from "@/lib/worker-client";
import type { FetchLyricsRequest, LyricsResult } from "@japanese-lyrics/shared";

/**
 * POST /api/lyrics
 * Triggers the lyrics provider chain via the Python worker.
 * Stores successful results as lyric lines in SQLite.
 */
export async function POST(req: Request) {
  const body = (await req.json()) as FetchLyricsRequest & { songId?: string };

  if (!body.youtubeId || !body.youtubeUrl) {
    return NextResponse.json(
      { error: "youtubeId and youtubeUrl are required" },
      { status: 400 }
    );
  }

  // Enrich with song metadata if we have it
  if (!body.title || !body.uploader) {
    const song = await prisma.song.findUnique({
      where: { youtubeId: body.youtubeId },
      select: { title: true, artist: true },
    });
    if (song) {
      body.title    = body.title    ?? song.title    ?? undefined;
      body.uploader = body.uploader ?? song.artist   ?? undefined;
    }
  }

  let result: LyricsResult;
  try {
    result = await workerClient.fetchLyrics(body);
  } catch (err) {
    return NextResponse.json(
      { error: `Worker error: ${err instanceof Error ? err.message : "unknown"}` },
      { status: 502 }
    );
  }

  // Persist lines to SQLite if we got lyrics
  if (result.lines.length > 0 && body.songId) {
    await prisma.lyricLine
      .deleteMany({ where: { songId: body.songId } })
      .catch(() => {});
    await prisma.lyricLine
      .createMany({
        data: result.lines.map((ln, i) => ({
          songId:    body.songId!,
          lineIndex: i,
          startTime: ln.startTime,
          endTime:   ln.endTime,
          japanese:  ln.text,
        })),
      })
      .catch(() => {});
  }

  return NextResponse.json(result);
}
