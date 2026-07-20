import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { AppSidebar } from "@/components/layout/AppSidebar";

export const metadata: Metadata = {
  title: "Adaptive RAG",
  description: "Adaptive RAG document QA console"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <Providers>
          <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
            <AppSidebar />
            <main className="min-w-0 p-4 lg:p-8">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
