import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { workerClient } from "@/lib/worker-client";
import { extractYoutubeId } from "@japanese-lyrics/shared";

export async function POST(req: Request) {
  const body = (await req.json()) as {
    songId?: string;
    youtubeUrl?: string;
    youtubeId?: string;
  };

  // Accept either a songId (existing song) or a raw youtubeUrl
  let songId = body.songId;
  let youtubeId = body.youtubeId;
  let youtubeUrl = body.youtubeUrl;

  if (!songId && !youtubeUrl) {
    return NextResponse.json(
      { error: "songId or youtubeUrl is required" },
      { status: 400 }
    );
  }

  // Resolve song record
  if (songId) {
    const song = await prisma.song.findUnique({ where: { id: songId } });
    if (!song) return NextResponse.json({ error: "Song not found" }, { status: 404 });
    youtubeId = song.youtubeId;
    youtubeUrl = song.youtubeUrl;
  } else {
    youtubeId = youtubeId ?? extractYoutubeId(youtubeUrl!) ?? undefined;
    if (!youtubeId) {
      return NextResponse.json({ error: "Invalid YouTube URL" }, { status: 400 });
    }
    const song = await prisma.song.upsert({
      where: { youtubeId },
      update: {},
      create: {
        youtubeUrl: youtubeUrl!,
        youtubeId,
        thumbnail: `https://i.ytimg.com/vi/${youtubeId}/mqdefault.jpg`,
      },
    });
    songId = song.id;
  }

  // Submit to Python worker
  let workerResp;
  try {
    workerResp = await workerClient.createJob({
      songId,
      youtubeId: youtubeId!,
      youtubeUrl: youtubeUrl!,
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Worker unavailable: ${err instanceof Error ? err.message : "unknown"}` },
      { status: 502 }
    );
  }

  // Persist jobId on the Song record
  await prisma.song.update({
    where: { id: songId },
    data: {
      currentJobId: workerResp.jobId,
      processingStatus: workerResp.status,
    },
  });

  return NextResponse.json(
    { jobId: workerResp.jobId, songId, status: workerResp.status },
    { status: 201 }
  );
}
