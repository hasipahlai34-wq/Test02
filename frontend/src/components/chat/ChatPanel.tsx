"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { ask } from "@/lib/api";
import { createRequestId, getOrCreateSessionId, resetSessionId } from "@/lib/session";
import { streamChat } from "@/lib/stream";
import type { SourceDoc, TraceSummary } from "@/lib/types";
import { useChatStore } from "@/store/chat-store";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageInput } from "./MessageInput";
import { MessageList } from "./MessageList";
import { SourcePanel } from "./SourcePanel";
import { TraceSummaryPanel } from "./TraceSummary";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === "AbortError";
}

export function ChatPanel() {
  const { messages, setMessages, trace, setTrace, sources, setSources, clear } = useChatStore();
  const [sessionId, setSessionId] = useState("");
  const [loading, setLoading] = useState(false);
  const messageViewportRef = useRef<HTMLDivElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setSessionId(getOrCreateSessionId());
  }, []);

  useEffect(() => {
    const viewport = messageViewportRef.current;
    if (!viewport) return;
    viewport.scrollTop = viewport.scrollHeight;
  }, [messages]);

  async function send(query: string) {
    const currentSession = sessionId || getOrCreateSessionId();
    const requestId = createRequestId();
    const userMessage = { id: requestId + "-user", role: "user" as const, content: query };
    const assistantMessage = { id: requestId + "-assistant", role: "assistant" as const, content: "" };
    setMessages([...messages, userMessage, assistantMessage]);
    setTrace({ session_id: currentSession, request_id: requestId });
    setSources([]);
    setLoading(true);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    const retrievalFilter = { session_id: currentSession };

    let streamedAnswer = "";
    const updateAssistant = (content: string) => {
      streamedAnswer = content;
      setMessages([...messages, userMessage, { ...assistantMessage, content }]);
    };
    const appendAssistant = (delta: string) => {
      streamedAnswer += delta;
      setMessages([...messages, userMessage, { ...assistantMessage, content: streamedAnswer }]);
    };

    try {
      await streamChat({ query, session_id: currentSession, request_id: requestId, retrieval_filter: retrievalFilter }, {
        signal: abortController.signal,
        onEvent: (event, data) => {
          if (event === "metadata" && isRecord(data)) {
            setTrace({ session_id: String(data.session_id ?? currentSession), request_id: String(data.request_id ?? requestId) });
          }
          if ((event === "state_update" || event === "done") && isRecord(data)) {
            setTrace({ ...(data as TraceSummary), session_id: currentSession, request_id: requestId });
          }
          if (event === "answer_delta" && isRecord(data)) {
            appendAssistant(String(data.text ?? ""));
          }
          if (event === "answer" && isRecord(data)) {
            updateAssistant(String(data.text ?? ""));
          }
          if (event === "sources" && isRecord(data) && Array.isArray(data.sources)) {
            setSources(data.sources as SourceDoc[]);
          }
          if (event === "error" && isRecord(data)) {
            throw new Error(String(data.message ?? "stream error"));
          }
        }
      });
    } catch (error) {
      if (isAbortError(error)) {
        const stoppedAnswer = streamedAnswer.trim() ? `${streamedAnswer}\n\n_已停止生成。_` : "_已停止生成。_";
        updateAssistant(stoppedAnswer);
        toast.info("已停止生成");
        return;
      }
      try {
        toast.warning("流式接口失败，已回退到 /ask。", { description: error instanceof Error ? error.message : String(error) });
        const fallback = await ask({ query, session_id: currentSession, request_id: requestId, retrieval_filter: retrievalFilter });
        updateAssistant(fallback.answer);
        setTrace({
          session_id: fallback.session_id,
          request_id: fallback.request_id,
          complexity: fallback.complexity,
          strategy: fallback.strategy,
          search_count: fallback.search_count,
          quality_score: fallback.quality_score
        });
      } catch (fallbackError) {
        const message = fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
        updateAssistant(`请求失败：${message}`);
        toast.error("问答失败", { description: message });
      }
    } finally {
      setLoading(false);
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
    }
  }

  function stopGenerating() {
    abortControllerRef.current?.abort();
  }

  function newSession() {
    stopGenerating();
    const id = resetSessionId();
    setSessionId(id);
    clear();
  }

  return (
    <div className="grid h-[calc(100vh-2rem)] min-h-0 gap-5 lg:h-[calc(100vh-4rem)] xl:grid-cols-[minmax(0,1fr)_380px]">
      <Card className="flex min-h-0 flex-col overflow-hidden">
        <CardHeader className="shrink-0 flex flex-row items-start justify-between gap-4">
          <div>
            <CardTitle>问答</CardTitle>
            <p className="mt-1 text-sm text-slate-500">通过 POST /chat/stream 接收 LangGraph workflow updates；失败时自动回退到 /ask。</p>
          </div>
          <Button variant="outline" onClick={newSession}>新会话</Button>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden">
          <div ref={messageViewportRef} className="min-h-0 flex-1 overflow-y-auto rounded-2xl bg-slate-50 p-4">
            <MessageList messages={messages} />
          </div>
          <div className="shrink-0 border-t pt-4">
            <MessageInput disabled={loading} generating={loading} onStop={stopGenerating} onSubmit={send} />
          </div>
        </CardContent>
      </Card>
      <aside className="min-h-0 space-y-5 overflow-y-auto">
        <TraceSummaryPanel trace={trace} />
        <SourcePanel sources={sources} />
      </aside>
    </div>
  );
}
