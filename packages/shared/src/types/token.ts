export interface Token {
  surface: string;
  reading: string | null;
  furigana: string | null;
  romaji: string | null;
  part_of_speech: string | null;
  dictionary_form: string | null;
  english: string | null;
  jlpt: "N1" | "N2" | "N3" | "N4" | "N5" | null;
  notes: string | null;
}

export interface GrammarPoint {
  pattern: string;
  meaning: string;
  jlpt: "N1" | "N2" | "N3" | "N4" | "N5" | null;
  example: string | null;
  notes: string | null;
}

export interface SentenceSegment {
  segment: string;
  function:
    | "topic"
    | "subject"
    | "object"
    | "predicate"
    | "reason"
    | "modifier"
    | "method"
    | "conjunction"
    | string;
  translation: string;
}

export interface DifficultyInfo {
  overall_jlpt: string;
  grammar_level: string;
  vocabulary_level: string;
  notes: string;
}

export interface AnalysisResult {
  translation: {
    literal: string;
    natural: string;
  };
  reading: {
    furigana: string;
    romaji: string;
  };
  tokens: Token[];
  grammar_points: GrammarPoint[];
  sentence_structure: SentenceSegment[];
  difficulty: DifficultyInfo;
}
