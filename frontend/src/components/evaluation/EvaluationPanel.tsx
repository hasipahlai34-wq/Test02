"use client";

import { useState } from "react";
import { toast } from "sonner";
import { runEvaluation } from "@/lib/api";
import { createRequestId, getOrCreateSessionId } from "@/lib/session";
import type { EvalResponse } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

function ResultBlock({ title, payload }: { title: string; payload?: Record<string, unknown> }) {
  const answer = payload?.answer ?? payload?.generated_answer ?? "-";
  return (
    <Card>
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="whitespace-pre-wrap rounded-xl bg-slate-50 p-3 text-sm leading-6">{String(answer)}</div>
        <pre className="max-h-72 overflow-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-100">{JSON.stringify(payload ?? {}, null, 2)}</pre>
      </CardContent>
    </Card>
  );
}

function ConclusionBlock({ conclusion }: { conclusion: string }) {
  const lines = String(conclusion || "-")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  return (
    <div className="space-y-2 text-sm leading-6 text-slate-700">
      {lines.map((line, index) => {
        const isHeading = line.startsWith("## ");
        const text = line.replace(/^##\s*/, "").replace(/^-\s*/, "");
        return (
          <div
            key={`${index}-${text}`}
            className={
              isHeading
                ? "text-base font-semibold text-slate-900"
                : "rounded-xl bg-slate-50 px-3 py-2"
            }
          >
            {text}
          </div>
        );
      })}
    </div>
  );
}

export function EvaluationPanel() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvalResponse | null>(null);

  async function submit() {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const response = await runEvaluation({ query, session_id: getOrCreateSessionId(), request_id: createRequestId() });
      setResult(response);
    } catch (error) {
      toast.error("评估失败", { description: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader>
          <CardTitle>三路对比评估</CardTitle>
          <p className="mt-1 text-sm text-slate-500">运行 direct answer、standard RAG、Adaptive RAG 并比较结果。</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea value={query} onChange={(event) => setQuery(event.target.value)} className="min-h-28" placeholder="输入一个需要比较的测试问题" />
          <Button onClick={submit} disabled={!query.trim() || loading}>{loading ? "运行中..." : "运行对比"}</Button>
        </CardContent>
      </Card>
      {result && (
        <>
          <Card>
            <CardHeader><CardTitle>评估结论</CardTitle></CardHeader>
            <CardContent><ConclusionBlock conclusion={result.conclusion} /></CardContent>
          </Card>
          <div className="grid gap-5 xl:grid-cols-3">
            <ResultBlock title="直接回答" payload={result.direct_answer} />
            <ResultBlock title="标准 RAG" payload={result.standard_rag} />
            <ResultBlock title="Adaptive RAG" payload={result.adaptive_rag} />
          </div>
        </>
      )}
    </div>
  );
}
