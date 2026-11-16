import type { ReactNode } from "react";

// Render a deliberately small, safe subset suitable for streamed tutoring text.
// React escapes all source text; raw HTML is never interpreted.
function inline(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|`[^`\n]+`)/g);
  return parts.map((part, i) => {
    if ((part.startsWith("**") && part.endsWith("**")) || (part.startsWith("__") && part.endsWith("__"))) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) return <em key={i}>{part.slice(1, -1)}</em>;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) return <code key={i} className="rounded bg-slate-100 px-1">{part.slice(1, -1)}</code>;
    return part;
  });
}

export function TutorMarkdown({ text }: { text: string }) {
  return <div className="text-sm leading-7">{text.split("\n").map((line, index) => {
    const heading = line.match(/^\s{0,3}#{1,6}\s+(.+)$/);
    if (heading) return <div key={index} role="heading" aria-level={3} className="mt-3 font-semibold text-slate-900">{inline(heading[1])}</div>;
    const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
    if (bullet) return <div key={index} className="flex gap-2 pl-2"><span aria-hidden="true">•</span><span>{inline(bullet[1])}</span></div>;
    const numbered = line.match(/^\s*(\d+[.)])\s+(.+)$/);
    if (numbered) return <div key={index} className="flex gap-2 pl-2"><span>{numbered[1]}</span><span>{inline(numbered[2])}</span></div>;
    return <div key={index} className="whitespace-pre-wrap min-h-[0.5em]">{inline(line)}</div>;
  })}</div>;
}
