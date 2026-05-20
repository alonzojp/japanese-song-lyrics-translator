import { NextResponse } from "next/server";
import { workerClient } from "@/lib/worker-client";

interface RouteParams {
  params: { youtubeId: string };
}

/** GET /api/lyrics/[youtubeId] — returns cached lyrics or 404 */
export async function GET(_req: Request, { params }: RouteParams) {
  try {
    const result = await workerClient.getLyrics(params.youtubeId);
    return NextResponse.json(result);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "unknown";
    if (msg.includes("404") || msg.includes("No cached")) {
      return NextResponse.json({ error: "No lyrics cached for this video" }, { status: 404 });
    }
    return NextResponse.json({ error: `Worker error: ${msg}` }, { status: 502 });
  }
}

/** POST /api/lyrics/[youtubeId]/manual — store user-provided lyrics text */
export async function POST(req: Request, { params }: RouteParams) {
  const body = (await req.json()) as { text?: string };
  if (!body.text?.trim()) {
    return NextResponse.json({ error: "text is required" }, { status: 400 });
  }
  try {
    const result = await workerClient.uploadManualLyrics(params.youtubeId, body.text);
    return NextResponse.json(result, { status: 201 });
  } catch (err) {
    return NextResponse.json(
      { error: `Worker error: ${err instanceof Error ? err.message : "unknown"}` },
      { status: 502 }
    );
  }
}
