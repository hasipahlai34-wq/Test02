"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "@/lib/types";

export function MessageList({ messages }: { messages: ChatMessage[] }) {
  if (!messages.length) {
    return <div className="rounded-2xl border border-dashed bg-white p-8 text-center text-sm text-slate-500">输入问题后，系统会展示回答、检索策略和来源。</div>;
  }
  return (
    <div className="space-y-4">
      {messages.map((message) => (
        <div key={message.id} className={message.role === "user" ? "flex justify-end" : "flex justify-start"}>
          <div className={message.role === "user" ? "max-w-[85%] rounded-2xl bg-slate-900 px-4 py-3 text-white" : "prose prose-slate max-w-[95%] rounded-2xl border bg-white px-4 py-3 shadow-sm"}>
            {message.role === "assistant" ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content || "…"}</ReactMarkdown> : message.content}
          </div>
        </div>
      ))}
    </div>
  );
}
