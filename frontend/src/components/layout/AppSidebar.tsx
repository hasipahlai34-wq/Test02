"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Database, MessageSquare, BarChart3, Wrench } from "lucide-react";
import { cn } from "@/lib/utils";

const nav = [
  { href: "/chat", label: "问答", icon: MessageSquare },
  { href: "/documents", label: "文档", icon: Database },
  { href: "/evaluation", label: "评估", icon: BarChart3 },
  { href: "/diagnostics", label: "诊断", icon: Wrench }
];

export function AppSidebar() {
  const pathname = usePathname();
  return (
    <aside className="border-b bg-white p-4 lg:min-h-screen lg:border-b-0 lg:border-r lg:p-6">
      <Link href="/" className="flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-900 text-white"><Activity className="h-5 w-5" /></span>
        <div>
          <div className="font-semibold">Adaptive RAG</div>
          <div className="text-xs text-slate-500">FastAPI + Next.js</div>
        </div>
      </Link>
      <nav className="mt-6 flex gap-2 overflow-x-auto lg:flex-col">
        {nav.map((item) => {
          const Icon = item.icon;
          const active = pathname === item.href;
          return (
            <Link key={item.href} href={item.href} className={cn("flex items-center gap-2 rounded-xl px-3 py-2 text-sm transition", active ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900")}>
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-6 rounded-xl border bg-slate-50 p-3 text-xs leading-5 text-slate-600">
        Langfuse 已占用 3000，因此前端默认运行在 3001。
      </div>
    </aside>
  );
}
