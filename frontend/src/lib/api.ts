import type { AskResponse, DiagnosticsResponse, EvalResponse, SessionDocumentsDeleteResponse, SessionDocumentsResponse, SourcesResponse, UploadResponse } from "./types";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {})
    }
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function health() {
  return requestJson<{ status: string; service: string }>("/health");
}

export function ask(payload: { query: string; session_id: string; request_id: string; user_id?: string; retrieval_filter?: Record<string, unknown> }) {
  return requestJson<AskResponse>("/ask", { method: "POST", body: JSON.stringify(payload) });
}

export function getSources() {
  return requestJson<SourcesResponse>("/sources");
}

export function getSessionDocuments(sessionId: string) {
  return requestJson<SessionDocumentsResponse>(`/sessions/${encodeURIComponent(sessionId)}/documents`);
}

export function clearSessionDocuments(sessionId: string) {
  return requestJson<SessionDocumentsDeleteResponse>(`/sessions/${encodeURIComponent(sessionId)}/documents`, { method: "DELETE" });
}

export function uploadDocuments(payload: { files: File[]; session_id: string; request_id: string; user_id?: string }) {
  const form = new FormData();
  payload.files.forEach((file) => form.append("files", file));
  form.append("session_id", payload.session_id);
  form.append("request_id", payload.request_id);
  if (payload.user_id) form.append("user_id", payload.user_id);
  return requestJson<UploadResponse>("/documents/upload", { method: "POST", body: form });
}

export function runEvaluation(payload: { query: string; session_id: string; request_id: string; user_id?: string }) {
  return requestJson<EvalResponse>("/eval", { method: "POST", body: JSON.stringify(payload) });
}

export function getDiagnostics() {
  return requestJson<DiagnosticsResponse>("/diagnostics");
}
