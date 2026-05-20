import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { workerClient } from "@/lib/worker-client";

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
      data: { processingStatus: job.status },
    })
    .catch(() => {});

  // When completed, store lyrics in DB
  if (job.status === "completed") {
    try {
      const result = await workerClient.getResult(jobId);
      const song = await prisma.song.findFirst({ where: { currentJobId: jobId } });

      if (song && result.lyrics.length > 0) {
        // Delete old lines then bulk-insert new ones
        await prisma.lyricLine.deleteMany({ where: { songId: song.id } });
        await prisma.lyricLine.createMany({
          data: result.lyrics.map((line) => ({
            songId: song.id,
            lineIndex: line.index,
            startTime: line.startTime,
            endTime: line.endTime,
            japanese: line.text,
            words: JSON.stringify(line.words),
          })),
        });
      }
    } catch {
      // Non-fatal — lyrics will be fetched on next poll
    }
  }

  return NextResponse.json(job);
}
