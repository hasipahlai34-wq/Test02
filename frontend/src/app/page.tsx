import Link from "next/link";
import { ArrowRight } from "lucide-react";

const cards = [
  { href: "/chat", title: "问答", desc: "自适应路由、检索、生成与来源追踪。" },
  { href: "/documents", title: "文档", desc: "上传文件并写入本地向量索引。" },
  { href: "/evaluation", title: "评估", desc: "直接回答、标准 RAG、Adaptive RAG 三路对比。" },
  { href: "/diagnostics", title: "诊断", desc: "查看 Langfuse、性能与 token 状态。" }
];

export default function HomePage() {
  return (
    <div className="space-y-8">
      <section className="rounded-2xl border bg-white p-8 shadow-sm">
        <p className="text-sm font-medium text-slate-500">Adaptive RAG Console</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight">文档问答与 RAG 评估控制台</h1>
        <p className="mt-3 max-w-3xl text-slate-600">
          新前端通过 FastAPI 调用后端，Langfuse tracing 在服务端完成。当前默认前端端口为 3001，避免与你的容器化 Langfuse 3000 端口冲突。
        </p>
      </section>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => (
          <Link key={card.href} href={card.href} className="rounded-2xl border bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md">
            <div className="flex items-center justify-between gap-4">
              <h2 className="text-lg font-semibold">{card.title}</h2>
              <ArrowRight className="h-4 w-4 text-slate-400" />
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-600">{card.desc}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
