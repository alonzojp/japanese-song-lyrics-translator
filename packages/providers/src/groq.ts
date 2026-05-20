import type { AnalysisResult } from "@japanese-lyrics/shared";
import { parseJSON } from "@japanese-lyrics/shared";

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const MODEL = "llama-3.3-70b-versatile";

const SYSTEM_PROMPT =
  "You are a Japanese language expert. Respond only with valid JSON. " +
  "CRITICAL RULE: every token \"reading\" field must be the pronunciation as actually " +
  "spoken in this sentence — never default to on-yomi when kun-yomi is used in context. " +
  "Example: 日 in メイドの日 = ひ (hi), not にち (nichi). 人 in 大人 = と (to), not じん (jin). " +
  "Always use context.";

const USER_PROMPT_PREFIX = `Analyze the following Japanese sentence and return ONLY a JSON object with this exact structure:
{
  "translation": { "literal": "word-for-word English", "natural": "natural English" },
  "reading": { "furigana": "full sentence with 漢字(reading) annotations", "romaji": "full romaji" },
  "tokens": [
    {
      "surface": "word as it appears",
      "reading": "hiragana reading (context-sensitive!)",
      "furigana": "食(た)べる format — annotate every kanji",
      "romaji": "hepburn romaji",
      "part_of_speech": "Noun/Verb/Adjective/Adverb/Particle/Conjunction/Interjection/Prefix/Suffix/Other",
      "dictionary_form": "base form or null if same",
      "english": "primary English meaning",
      "jlpt": "N5/N4/N3/N2/N1 or null",
      "notes": "usage note or null"
    }
  ],
  "grammar_points": [
    {
      "pattern": "〜pattern",
      "meaning": "English meaning",
      "jlpt": "N5/N4/N3/N2/N1",
      "example": "excerpt from the sentence",
      "notes": "nuance or usage"
    }
  ],
  "sentence_structure": [
    {
      "segment": "chunk of the sentence",
      "function": "topic | subject | object | predicate | reason | modifier | method | conjunction",
      "translation": "English for this chunk"
    }
  ],
  "difficulty": {
    "overall_jlpt": "highest JLPT level needed",
    "grammar_level": "e.g. N5-N3",
    "vocabulary_level": "e.g. Mixed (N5 to N1)",
    "notes": "one-sentence summary of difficulty"
  }
}

Sentence: `;

export async function analyzeWithGroq(
  text: string,
  apiKey: string
): Promise<AnalysisResult> {
  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: USER_PROMPT_PREFIX + text },
      ],
      temperature: 0.1,
      response_format: { type: "json_object" },
    }),
  });

  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
    throw new Error(err.error?.message ?? `Groq API error ${res.status}`);
  }

  const data = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const content = data.choices?.[0]?.message?.content;
  if (!content) throw new Error("Empty response from Groq");

  return parseJSON<AnalysisResult>(content);
}
