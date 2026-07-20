import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TraceSummary } from "@/lib/types";

function fmt(value: unknown) {
  if (value === undefined || value === null || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

export function TraceSummaryPanel({ trace }: { trace: TraceSummary }) {
  const rows = [
    ["节点", trace.node],
    ["复杂度", trace.complexity],
    ["策略", trace.strategy],
    ["检索命中", trace.search_count],
    ["质量分", trace.quality_score],
    ["缓存命中", trace.cache_hit],
    ["质量通过", trace.quality_passed],
    ["安全风险", trace.safety_risk_level],
    ["人工审核", trace.hitl_status]
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>本次链路</CardTitle>
        <div className="mt-2 flex flex-wrap gap-2">
          {trace.session_id && <Badge>session: {trace.session_id.slice(0, 18)}…</Badge>}
          {trace.request_id && <Badge>request: {trace.request_id.slice(0, 8)}…</Badge>}
        </div>
      </CardHeader>
      <CardContent>
        <dl className="space-y-2 text-sm">
          {rows.map(([key, value]) => (
            <div key={key as string} className="flex justify-between gap-4 border-b pb-2 last:border-b-0">
              <dt className="text-slate-500">{key}</dt>
              <dd className="font-medium text-slate-900">{fmt(value)}</dd>
            </div>
          ))}
        </dl>
        {trace.ragas_scores && Object.keys(trace.ragas_scores).length > 0 && (
          <pre className="mt-4 max-h-52 overflow-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-100">{JSON.stringify(trace.ragas_scores, null, 2)}</pre>
        )}
      </CardContent>
    </Card>
  );
}
