import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { KaraokePlayer } from "@/components/karaoke-player";
import { prisma } from "@/lib/prisma";
import type { LyricLine, WordTiming } from "@japanese-lyrics/shared";

interface PlayerPageProps {
  params: { id: string };
}

async function getSong(id: string) {
  const song = await prisma.song.findUnique({
    where: { id },
    include: { lyrics: { orderBy: { lineIndex: "asc" } } },
  });
  if (!song) return null;

  const lyrics: LyricLine[] = song.lyrics.map((line) => ({
    id:        `line-${line.lineIndex}`,
    text:      line.japanese,
    startTime: line.startTime ?? 0,
    endTime:   line.endTime   ?? 0,
    words:     line.words ? (JSON.parse(line.words) as WordTiming[]) : undefined,
  }));

  return {
    id:               song.id,
    youtubeId:        song.youtubeId,
    title:            song.title,
    artist:           song.artist,
    currentJobId:     song.currentJobId,
    processingStatus: song.processingStatus,
    lyrics,
  };
}

export async function generateMetadata({ params }: PlayerPageProps): Promise<Metadata> {
  const song = await getSong(params.id);
  if (!song) return { title: "Not Found" };
  return {
    title:       song.title ?? "Song Player",
    description: song.artist ? `${song.title} by ${song.artist}` : (song.title ?? undefined),
  };
}

export default async function PlayerPage({ params }: PlayerPageProps) {
  const song = await getSong(params.id);
  if (!song) notFound();

  return (
    <main className="mx-auto max-w-4xl px-4 py-8">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to home
      </Link>

      <KaraokePlayer
        songId={song.id}
        youtubeId={song.youtubeId}
        title={song.title ?? null}
        artist={song.artist ?? null}
        initialJobId={song.currentJobId ?? null}
        initialStatus={song.processingStatus ?? null}
        initialLyrics={song.lyrics}
      />
    </main>
  );
}
