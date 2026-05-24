import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { workerClient } from "@/lib/worker-client";
import type { WordTiming } from "@japanese-lyrics/shared";

interface RouteParams {
  params: { id: string };
}

/** Poll job status from the Python worker and sync it to our DB. */
export async function GET(_req: Request, { params }: RouteParams) {
  const jobId = params.id;

  let job;
  try {
    job = await workerClient.getJob(jobId);
  } catch (err) {
    return NextResponse.json(
      { error: `Worker unavailable: ${err instanceof Error ? err.message : "unknown"}` },
      { status: 502 }
    );
  }

  // Keep Song.processingStatus in sync
  await prisma.song
    .updateMany({
      where: { currentJobId: jobId },
      data:  { processingStatus: job.status },
    })
    .catch(() => {});

  // When completed, persist lyrics + word timings to DB
  if (job.status === "completed") {
    try {
      const result = await workerClient.getResult(jobId);
      const song   = await prisma.song.findFirst({ where: { currentJobId: jobId } });

      if (song) {
        // Prefer aligned_lines (LyricLine format with word timings) over legacy lyrics
        const alignedLines = (result as unknown as Record<string, unknown>).alignedLines as
          Array<{
            id: string;
            text: string;
            startTime: number;
            endTime: number;
            acousticStart?: number;
            acousticEnd?: number;
            words?: WordTiming[];
            confidence?: number;
          }> | undefined;

        const lines = alignedLines ?? result.lyrics.map((l, i) => ({
          id:        `line-${i}`,
          text:      l.text,
          startTime: l.startTime,
          endTime:   l.endTime,
          words:     l.words?.map((w) => ({ word: w.word, start: w.startTime, end: w.endTime })),
        }));

        if (lines.length > 0) {
          await prisma.lyricLine.deleteMany({ where: { songId: song.id } });
          await prisma.lyricLine.createMany({
            data: lines.map((line, i) => ({
              songId:       song.id,
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
      }
    } catch {
      // Non-fatal — retry on next poll
    }
  }

  return NextResponse.json(job);
}
