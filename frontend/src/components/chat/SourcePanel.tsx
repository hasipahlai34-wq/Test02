import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { SourceDoc } from "@/lib/types";

export function SourcePanel({ sources }: { sources: SourceDoc[] }) {
  return (
    <Card>
      <CardHeader><CardTitle>参考来源</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        {!sources.length && <p className="text-sm text-slate-500">暂无来源。</p>}
        {sources.map((source, index) => (
          <div key={`${source.source}-${index}`} className="rounded-xl border bg-slate-50 p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="font-medium">{index + 1}. {source.title}</div>
              <Badge>{source.source}</Badge>
            </div>
            <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-600">{source.content}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
