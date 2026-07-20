"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { clearSessionDocuments, getSessionDocuments, getSources, uploadDocuments } from "@/lib/api";
import { createRequestId, getOrCreateSessionId } from "@/lib/session";
import type { UploadResult } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

function formatQuality(score?: number | null) {
  if (score == null) return "-";
  return `${Math.round(score * 100)}%`;
}

function formatUploadedAt(value?: string | null) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString();
}

export function DocumentsPanel() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [lastUploadResults, setLastUploadResults] = useState<UploadResult[]>([]);
  const queryClient = useQueryClient();

  useEffect(() => {
    setSessionId(getOrCreateSessionId());
  }, []);

  const sources = useQuery({ queryKey: ["sources"], queryFn: getSources });
  const documents = useQuery({
    queryKey: ["session-documents", sessionId],
    queryFn: () => getSessionDocuments(sessionId),
    enabled: Boolean(sessionId)
  });

  const persistedResults = documents.data?.documents ?? [];
  const failedResults = lastUploadResults.filter((result) => result.status === "error");
  const optimisticResults = lastUploadResults.filter(
    (result) => result.status === "ok" && !persistedResults.some((item) => item.document_id === result.document_id)
  );
  const tableRows = [...failedResults, ...optimisticResults, ...persistedResults];

  async function refreshDocuments(currentSession = sessionId) {
    await Promise.all([
      sources.refetch(),
      currentSession ? queryClient.invalidateQueries({ queryKey: ["session-documents", currentSession] }) : Promise.resolve()
    ]);
  }

  async function submit() {
    if (!files.length) return;
    const currentSession = sessionId || getOrCreateSessionId();
    setSessionId(currentSession);
    setUploading(true);
    try {
      const response = await uploadDocuments({
        files,
        session_id: currentSession,
        request_id: createRequestId()
      });
      setLastUploadResults(response.results);
      toast.success(`索引完成：${response.total_indexed} chunks`);
      await refreshDocuments(currentSession);
    } catch (error) {
      toast.error("上传失败", { description: error instanceof Error ? error.message : String(error) });
    } finally {
      setUploading(false);
    }
  }

  async function clearCurrentSessionDocuments() {
    if (!sessionId || persistedResults.length === 0) return;
    const confirmed = window.confirm("确认清空当前会话的已索引文档吗？该操作不会删除其他会话的文档。");
    if (!confirmed) return;

    setClearing(true);
    try {
      const response = await clearSessionDocuments(sessionId);
      setLastUploadResults([]);
      toast.success(`已清空当前会话文档：${response.deleted} chunks`);
      await refreshDocuments(sessionId);
    } catch (error) {
      toast.error("清空失败", { description: error instanceof Error ? error.message : String(error) });
    } finally {
      setClearing(false);
    }
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
      <Card>
        <CardHeader>
          <CardTitle>文档摄入</CardTitle>
          <p className="mt-1 text-sm text-slate-500">浏览器通过 multipart/form-data 上传文件，后端写入当前会话的本地索引。</p>
        </CardHeader>
        <CardContent className="space-y-5">
          <Input type="file" multiple accept=".txt,.md,.pdf,.docx,.csv" onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
          <div className="rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm text-blue-900">
            系统会自动解析 PDF / DOCX / Markdown / TXT / CSV 的通用结构，生成文档大纲、结构元素和动态分块计划；用户无需选择分块策略或大小。
          </div>
          <Button onClick={submit} disabled={!files.length || uploading}>{uploading ? "处理中..." : "开始处理文档"}</Button>

          <div className="flex items-center justify-between gap-3">
            <div className="text-sm text-slate-500">
              当前会话已索引文档：{documents.isLoading ? "加载中..." : persistedResults.length}
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => refreshDocuments()} disabled={!sessionId || documents.isFetching}>刷新文档状态</Button>
              <Button variant="outline" onClick={clearCurrentSessionDocuments} disabled={!sessionId || clearing || persistedResults.length === 0}>清空当前会话文档</Button>
            </div>
          </div>

          {tableRows.length > 0 ? (
            <div className="overflow-x-auto rounded-xl border">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-left text-slate-500">
                  <tr>
                    <th className="p-3">文件</th>
                    <th className="p-3">状态</th>
                    <th className="p-3">文档ID</th>
                    <th className="p-3">上传时间</th>
                    <th className="p-3">解析质量</th>
                    <th className="p-3">结构元素</th>
                    <th className="p-3">自动策略</th>
                    <th className="p-3">目标大小</th>
                    <th className="p-3">Chunks</th>
                    <th className="p-3">写入</th>
                    <th className="p-3">错误</th>
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((result, index) => (
                    <tr key={`${result.document_id ?? result.filename}-${index}`} className="border-t align-top">
                      <td className="min-w-64 p-3">
                        <div className="font-medium text-slate-900">{result.filename}</div>
                        {result.outline_preview && (
                          <div className="mt-1 max-w-xl whitespace-pre-wrap text-xs text-slate-500">
                            {result.outline_preview.slice(0, 180)}
                            {result.outline_preview.length > 180 ? "..." : ""}
                          </div>
                        )}
                      </td>
                      <td className="p-3">{result.status}</td>
                      <td className="p-3 font-mono text-xs">{result.document_id ? result.document_id.slice(0, 18) : "-"}</td>
                      <td className="p-3 whitespace-nowrap">{formatUploadedAt(result.uploaded_at)}</td>
                      <td className="p-3">{formatQuality(result.parse_quality_score)}</td>
                      <td className="p-3">
                        {result.element_count ?? "-"}
                        {result.warning_count ? <span className="ml-1 text-amber-600">({result.warning_count} warnings)</span> : null}
                      </td>
                      <td className="p-3">{result.chunk_strategy ?? "-"}</td>
                      <td className="p-3">{result.target_tokens ? `${result.target_tokens} tokens` : "-"}</td>
                      <td className="p-3">{result.chunks}</td>
                      <td className="p-3">{result.indexed}</td>
                      <td className="p-3 text-red-600">{result.error}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-xl border border-dashed bg-slate-50 p-6 text-sm text-slate-500">当前会话暂无已索引文档。</div>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-3"><CardTitle>知识库来源</CardTitle><Button variant="outline" onClick={() => refreshDocuments()}>刷新</Button></CardHeader>
        <CardContent>
          <div className="mb-3 text-sm text-slate-500">全局已索引 chunks：{sources.data?.total_chunks ?? "-"}</div>
          <div className="max-h-[640px] space-y-2 overflow-auto">
            {sources.isLoading && <p className="text-sm text-slate-500">加载中...</p>}
            {sources.data?.sources?.map((source) => <div key={source} className="rounded-xl border bg-slate-50 p-3 text-sm">{source}</div>)}
            {sources.data && sources.data.sources.length === 0 && <p className="text-sm text-slate-500">暂无来源。</p>}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
