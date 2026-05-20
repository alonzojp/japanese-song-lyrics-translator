import { NextRequest, NextResponse } from "next/server";
import type { ExportCard } from "@japanese-lyrics/anki";
import { cardsToAnkiTsv } from "@japanese-lyrics/anki";
import { buildApkg } from "@/lib/apkg-builder";

export async function POST(req: NextRequest) {
  let body: { cards: ExportCard[]; deckName?: string; format?: "apkg" | "csv" };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { cards, deckName = "Japanese Lyrics", format = "apkg" } = body;

  if (!Array.isArray(cards) || cards.length === 0) {
    return NextResponse.json({ error: "No cards provided" }, { status: 400 });
  }

  if (format === "csv") {
    const tsv = cardsToAnkiTsv(cards, deckName);
    const bom = "﻿";
    return new NextResponse(bom + tsv, {
      status: 200,
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Disposition": `attachment; filename="${sanitizeFilename(deckName)}.txt"`,
      },
    });
  }

  // Default: apkg
  let apkgBytes: Uint8Array;
  try {
    apkgBytes = await buildApkg(cards, deckName);
  } catch (err) {
    console.error("[anki/export] buildApkg failed:", err);
    return NextResponse.json({ error: "Failed to build .apkg" }, { status: 500 });
  }

  return new NextResponse(apkgBytes.buffer as ArrayBuffer, {
    status: 200,
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": `attachment; filename="${sanitizeFilename(deckName)}.apkg"`,
    },
  });
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^\w\s\-.]/g, "_").replace(/\s+/g, "_").slice(0, 80);
}
