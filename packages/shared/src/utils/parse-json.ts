export function parseJSON<T = unknown>(text: string): T {
  try {
    return JSON.parse(text) as T;
  } catch {}

  const codeBlock = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlock?.[1]) {
    try {
      return JSON.parse(codeBlock[1]) as T;
    } catch {}
  }

  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start !== -1 && end !== -1) {
    try {
      return JSON.parse(text.slice(start, end + 1)) as T;
    } catch {}
  }

  throw new Error("Could not parse AI response as JSON");
}
