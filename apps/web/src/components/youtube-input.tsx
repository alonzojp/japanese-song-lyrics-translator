"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Music, Youtube } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { extractYoutubeId } from "@japanese-lyrics/shared";

export function YouTubeInput() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    const youtubeId = extractYoutubeId(url.trim());
    if (!youtubeId) {
      setError("Please enter a valid YouTube URL.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("/api/songs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtubeUrl: url.trim(), youtubeId }),
      });

      if (!res.ok) {
        const data = (await res.json()) as { error?: string };
        throw new Error(data.error ?? "Failed to add song");
      }

      const song = (await res.json()) as { id: string };
      router.push(`/player/${song.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl space-y-3">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Youtube className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="url"
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              setError(null);
            }}
            className="pl-10 text-base"
            disabled={loading}
            aria-label="YouTube URL"
          />
        </div>
        <Button type="submit" disabled={loading || !url.trim()} size="lg">
          {loading ? (
            <>
              <Music className="animate-pulse" />
              Loading...
            </>
          ) : (
            <>
              <Music />
              Translate
            </>
          )}
        </Button>
      </div>

      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}

      <p className="text-center text-xs text-muted-foreground">
        Paste any Japanese song URL from YouTube to get started
      </p>
    </form>
  );
}
