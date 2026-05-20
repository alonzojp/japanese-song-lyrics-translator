export interface TimedLine {
  index: number;
  startTime: number;
  endTime: number;
  text: string;
}

export interface AlignmentResult {
  lines: TimedLine[];
  confidence: number;
  method: "whisper" | "forced" | "manual" | "placeholder";
}

export interface AlignmentOptions {
  language?: string;
  model?: string;
}
