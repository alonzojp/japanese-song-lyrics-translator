import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { workerClient } from "@/lib/worker-client";

interface RouteParams {
  params: { id: string };
}

/** Return the most log-rich alignment run for this song — bypasses currentJobId. */
export async function GET(_req: Request, { params }: RouteParams) {
  const song = await prisma.song.findUnique({ where: { id: params.id } });
  if (!song) return NextResponse.json({ error: "Song not found" }, { status: 404 });

  try {
    const result = await workerClient.getBestLogs(song.youtubeId);
    return NextResponse.json(result);
  } catch {
    return NextResponse.json({ youtubeId: song.youtubeId, logCount: 0, logs: [] });
  }
}
