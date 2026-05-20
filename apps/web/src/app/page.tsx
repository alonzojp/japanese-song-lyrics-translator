import Link from "next/link";
import Image from "next/image";
import { Music2, BookOpen, Zap, ChevronRight } from "lucide-react";
import { YouTubeInput } from "@/components/youtube-input";
import { Card, CardContent } from "@/components/ui/card";
import { prisma } from "@/lib/prisma";

async function getRecentSongs() {
  try {
    return await prisma.song.findMany({
      orderBy: { createdAt: "desc" },
      take: 6,
      select: {
        id: true,
        youtubeId: true,
        title: true,
        artist: true,
        thumbnail: true,
      },
    });
  } catch {
    return [];
  }
}

export default async function HomePage() {
  const recentSongs = await getRecentSongs();

  return (
    <main className="flex min-h-screen flex-col items-center">
      {/* Hero */}
      <section className="flex w-full flex-col items-center gap-8 px-4 pb-16 pt-24">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-primary/10 p-3">
              <Music2 className="h-8 w-8 text-primary" />
            </div>
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
              Japanese Song{" "}
              <span className="text-primary">Lyrics</span>{" "}
              Translator
            </h1>
          </div>
          <p className="max-w-xl text-lg text-muted-foreground">
            Learn Japanese through music. Paste a YouTube URL to get furigana readings,
            word-by-word translations, and karaoke-style lyric highlighting.
          </p>
        </div>

        <YouTubeInput />

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-3 text-sm text-muted-foreground">
          {[
            { icon: BookOpen, label: "Furigana & Romaji" },
            { icon: Zap, label: "AI-powered analysis" },
            { icon: Music2, label: "Karaoke highlighting" },
          ].map(({ icon: Icon, label }) => (
            <span key={label} className="flex items-center gap-1.5 rounded-full border px-3 py-1">
              <Icon className="h-3.5 w-3.5 text-primary" />
              {label}
            </span>
          ))}
        </div>
      </section>

      {/* Recent songs */}
      {recentSongs.length > 0 && (
        <section className="w-full max-w-4xl px-4 pb-16">
          <h2 className="mb-6 text-xl font-semibold">Recent Songs</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {recentSongs.map((song) => (
              <Link key={song.id} href={`/player/${song.id}`}>
                <Card className="group cursor-pointer transition-colors hover:border-primary/50">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      {song.thumbnail ? (
                        <div className="relative h-14 w-14 shrink-0 overflow-hidden rounded-lg">
                          <Image
                            src={song.thumbnail}
                            alt={song.title ?? "Song thumbnail"}
                            fill
                            className="object-cover"
                            sizes="56px"
                          />
                        </div>
                      ) : (
                        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-muted">
                          <Music2 className="h-5 w-5 text-muted-foreground" />
                        </div>
                      )}
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium">
                          {song.title ?? "Unknown Title"}
                        </p>
                        {song.artist && (
                          <p className="truncate text-sm text-muted-foreground">{song.artist}</p>
                        )}
                      </div>
                      <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
