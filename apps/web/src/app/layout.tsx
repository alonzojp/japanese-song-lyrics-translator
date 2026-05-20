import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Japanese Song Lyrics Translator",
    template: "%s | Japanese Song Lyrics Translator",
  },
  description:
    "Learn Japanese through music. Get furigana, translations, and word-by-word breakdowns for Japanese song lyrics with karaoke-style highlighting.",
  keywords: ["Japanese", "lyrics", "translation", "furigana", "karaoke", "language learning"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}>
        <div className="min-h-screen">{children}</div>
      </body>
    </html>
  );
}
