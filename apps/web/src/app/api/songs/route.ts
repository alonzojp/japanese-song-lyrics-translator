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
      processingStatus: true,
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

  // Fetch title + thumbnail via YouTube oEmbed (free, no API key needed)
  const meta = await fetchOEmbed(rawUrl);

  const song = await prisma.song.upsert({
    where: { youtubeId },
    update: {
      // Refresh metadata if we got better data
      ...(meta.title     && { title: meta.title }),
      ...(meta.thumbnail && { thumbnail: meta.thumbnail }),
    },
    create: {
      youtubeUrl: rawUrl,
      youtubeId,
      title:     meta.title     ?? null,
      artist:    meta.author    ?? null,
      thumbnail: meta.thumbnail ?? `https://i.ytimg.com/vi/${youtubeId}/mqdefault.jpg`,
    },
  });

  return NextResponse.json(song, { status: 201 });
}

// ── YouTube oEmbed ─────────────────────────────────────────────────────────────

interface OEmbedResponse {
  title?: string;
  author_name?: string;
  thumbnail_url?: string;
}

async function fetchOEmbed(
  youtubeUrl: string
): Promise<{ title: string | null; author: string | null; thumbnail: string | null }> {
  try {
    const endpoint =
      `https://www.youtube.com/oembed?url=${encodeURIComponent(youtubeUrl)}&format=json`;
    const res = await fetch(endpoint, {
      next: { revalidate: 86400 }, // cache for 24 h (Next.js fetch cache)
    });
    if (!res.ok) return { title: null, author: null, thumbnail: null };
    const data = (await res.json()) as OEmbedResponse;
    return {
      title:     data.title        ?? null,
      author:    data.author_name  ?? null,
      thumbnail: data.thumbnail_url ?? null,
    };
  } catch {
    return { title: null, author: null, thumbnail: null };
  }
}
