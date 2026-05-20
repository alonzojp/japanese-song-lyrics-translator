import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { extractYoutubeId } from "@japanese-lyrics/shared";

export async function GET() {
  const songs = await prisma.song.findMany({
    orderBy: { createdAt: "desc" },
    select: {
      id: true,
      youtubeUrl: true,
      youtubeId: true,
      title: true,
      artist: true,
      thumbnail: true,
      createdAt: true,
      _count: { select: { lyrics: true } },
    },
  });
  return NextResponse.json(songs);
}

export async function POST(req: Request) {
  const body = (await req.json()) as { youtubeUrl?: string; youtubeId?: string };

  const rawUrl = body.youtubeUrl?.trim();
  if (!rawUrl) {
    return NextResponse.json({ error: "youtubeUrl is required" }, { status: 400 });
  }

  const youtubeId = body.youtubeId ?? extractYoutubeId(rawUrl);
  if (!youtubeId) {
    return NextResponse.json({ error: "Invalid YouTube URL" }, { status: 400 });
  }

  // Upsert so re-submitting the same URL just returns the existing record
  const song = await prisma.song.upsert({
    where: { youtubeId },
    update: {},
    create: {
      youtubeUrl: rawUrl,
      youtubeId,
      // Thumbnail from YouTube's public image CDN (no API key required)
      thumbnail: `https://i.ytimg.com/vi/${youtubeId}/mqdefault.jpg`,
    },
  });

  return NextResponse.json(song, { status: 201 });
}
