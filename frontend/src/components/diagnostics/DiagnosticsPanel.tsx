"use client";

import { useQuery } from "@tanstack/react-query";
import { getDiagnostics, health } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function DiagnosticsPanel() {
  const diagnostics = useQuery({ queryKey: ["diagnostics"], queryFn: getDiagnostics });
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: health });
  const data = diagnostics.data;
  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">运行诊断</h1>
          <p className="mt-1 text-sm text-slate-500">这里不会展示任何 Langfuse、OpenAI 或后端密钥。</p>
        </div>
        <Button variant="outline" onClick={() => { diagnostics.refetch(); healthQuery.refetch(); }}>刷新</Button>
      </div>
      <div className="grid gap-5 xl:grid-cols-3">
        <Card>
          <CardHeader><CardTitle>服务状态</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm">
            <Badge>{healthQuery.data?.status ?? "unknown"}</Badge>
            <pre className="rounded-xl bg-slate-950 p-3 text-xs text-slate-100">{JSON.stringify(healthQuery.data ?? {}, null, 2)}</pre>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Langfuse</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div>启用：{data?.langfuse.enabled ? "是" : "否"}</div>
            <div>已配置：{data?.langfuse.configured ? "是" : "否"}</div>
            <div>地址：{data?.langfuse.base_url ?? "-"}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Token</CardTitle></CardHeader>
          <CardContent><pre className="max-h-72 overflow-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-100">{JSON.stringify(data?.tokens ?? {}, null, 2)}</pre></CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader><CardTitle>性能统计</CardTitle></CardHeader>
        <CardContent><pre className="max-h-[520px] overflow-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-100">{JSON.stringify(data?.performance ?? {}, null, 2)}</pre></CardContent>
      </Card>
    </div>
  );
}
