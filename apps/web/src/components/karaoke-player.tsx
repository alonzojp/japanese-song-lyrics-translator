"use client";

import { useState } from "react";
import { Play, Pause, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Song } from "@japanese-lyrics/shared";

interface KaraokePlayerProps {
  song: Pick<Song, "id" | "youtubeId" | "title" | "artist" | "lyrics">;
}

export function KaraokePlayer({ song }: KaraokePlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime] = useState(0);

  const activeLine = song.lyrics.find(
    (line) =>
      line.startTime !== null &&
      line.endTime !== null &&
      currentTime >= line.startTime &&
      currentTime <= line.endTime
  );

  return (
    <div className="flex flex-col gap-6">
      {/* YouTube embed placeholder */}
      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-muted">
        <iframe
          src={`https://www.youtube.com/embed/${song.youtubeId}?enablejsapi=1`}
          title={song.title ?? "YouTube video"}
          className="absolute inset-0 h-full w-full"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>

      {/* Song info */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">{song.title ?? "Unknown Title"}</h1>
          {song.artist && <p className="text-muted-foreground">{song.artist}</p>}
        </div>
        <Badge variant="outline" className="gap-1">
          <Clock className="h-3 w-3" />
          Sync pending
        </Badge>
      </div>

      {/* Karaoke lyrics display */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Lyrics</CardTitle>
        </CardHeader>
        <CardContent>
          {song.lyrics.length === 0 ? (
            <div className="flex flex-col items-center gap-3 py-12 text-center">
              <div className="rounded-full bg-muted p-4">
                <Play className="h-6 w-6 text-muted-foreground" />
              </div>
              <div>
                <p className="font-medium">No lyrics yet</p>
                <p className="text-sm text-muted-foreground">
                  Lyrics will appear here once processed
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={() => setIsPlaying(!isPlaying)}>
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isPlaying ? "Pause" : "Play"}
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              {song.lyrics.map((line) => (
                <div
                  key={line.index}
                  className={`rounded-lg px-4 py-3 transition-all duration-300 ${
                    activeLine?.index === line.index
                      ? "animate-karaoke-highlight bg-primary/10 text-primary"
                      : "text-foreground/70 hover:bg-muted"
                  }`}
                >
                  <p
                    className="font-japanese text-lg leading-loose"
                    dangerouslySetInnerHTML={{ __html: line.japanese }}
                  />
                  {line.analysis?.translation.natural && (
                    <p className="mt-1 text-sm text-muted-foreground">
                      {line.analysis.translation.natural}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Synchronization notice */}
      <Card className="border-dashed">
        <CardContent className="flex items-center gap-3 py-4">
          <Clock className="h-4 w-4 shrink-0 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            <span className="font-medium text-foreground">Karaoke synchronization coming soon.</span>{" "}
            Lyrics will automatically highlight in sync with the video once timing alignment is
            implemented.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
