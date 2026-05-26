import { NextResponse } from "next/server";
import { workerClient } from "@/lib/worker-client";

export async function DELETE(
  _req: Request,
  { params }: { params: { youtubeId: string } },
) {
  try {
    const result = await workerClient.invalidateForceAlign(params.youtubeId);
    return NextResponse.json(result);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    const status = msg.includes("404") ? 404 : 502;
    return NextResponse.json({ error: msg }, { status });
  }
}
