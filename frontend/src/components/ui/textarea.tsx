import * as React from "react";
import { cn } from "@/lib/utils";

export function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={cn("w-full rounded-xl border bg-white px-3 py-2 text-sm outline-none ring-slate-300 transition focus:ring-2", props.className)} />;
}
