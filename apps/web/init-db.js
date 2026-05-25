// Runs at container startup to ensure SQLite schema exists.
// Uses raw SQL so it works without prisma CLI or migrations.
const { PrismaClient } = require("@prisma/client");
const prisma = new PrismaClient();

async function main() {
  await prisma.$executeRawUnsafe(`
    CREATE TABLE IF NOT EXISTS "Song" (
      "id" TEXT NOT NULL PRIMARY KEY,
      "youtubeUrl" TEXT NOT NULL,
      "youtubeId" TEXT NOT NULL,
      "title" TEXT,
      "artist" TEXT,
      "thumbnail" TEXT,
      "currentJobId" TEXT,
      "processingStatus" TEXT,
      "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      "updatedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
  `);
  await prisma.$executeRawUnsafe(
    `CREATE UNIQUE INDEX IF NOT EXISTS "Song_youtubeUrl_key" ON "Song"("youtubeUrl")`
  );
  await prisma.$executeRawUnsafe(
    `CREATE UNIQUE INDEX IF NOT EXISTS "Song_youtubeId_key" ON "Song"("youtubeId")`
  );
  await prisma.$executeRawUnsafe(`
    CREATE TABLE IF NOT EXISTS "LyricLine" (
      "id" TEXT NOT NULL PRIMARY KEY,
      "songId" TEXT NOT NULL,
      "lineIndex" INTEGER NOT NULL,
      "startTime" REAL,
      "endTime" REAL,
      "japanese" TEXT NOT NULL,
      "words" TEXT,
      "tokens" TEXT,
      "analysis" TEXT,
      "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY ("songId") REFERENCES "Song"("id") ON DELETE CASCADE
    )
  `);
  await prisma.$executeRawUnsafe(
    `CREATE INDEX IF NOT EXISTS "LyricLine_songId_lineIndex_idx" ON "LyricLine"("songId", "lineIndex")`
  );

  // Backfill columns added after initial schema (SQLite has no IF NOT EXISTS for columns).
  for (const col of [
    `ALTER TABLE "LyricLine" ADD COLUMN "acousticStart" REAL`,
    `ALTER TABLE "LyricLine" ADD COLUMN "acousticEnd"   REAL`,
  ]) {
    try { await prisma.$executeRawUnsafe(col); } catch (_) { /* column already exists */ }
  }

  await prisma.$disconnect();
  console.log("Database initialized");
}

main().catch((e) => {
  console.error("DB init failed:", e);
  process.exit(1);
});
