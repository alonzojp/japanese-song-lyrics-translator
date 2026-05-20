import type { Token, AnalysisResult } from "./token.js";

export interface LyricLine {
  index: number;
  startTime: number | null;
  endTime: number | null;
  japanese: string;
  tokens: Token[] | null;
  analysis: AnalysisResult | null;
}

export interface Song {
  id: string;
  youtubeUrl: string;
  youtubeId: string;
  title: string | null;
  artist: string | null;
  thumbnail: string | null;
  lyrics: LyricLine[];
  createdAt: Date;
  updatedAt: Date;
}

export function extractYoutubeId(url: string): string | null {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
    /youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})/,
  ];
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match?.[1]) return match[1];
  }
  return null;
}
