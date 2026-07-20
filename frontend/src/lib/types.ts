export type SourceDoc = {
  title: string;
  source: string;
  content: string;
  metadata: Record<string, unknown>;
};

export type TraceSummary = {
  session_id?: string;
  request_id?: string;
  node?: string;
  complexity?: string | null;
  strategy?: string | null;
  search_count?: number | null;
  quality_score?: number | null;
  cache_hit?: boolean | null;
  quality_passed?: boolean | null;
  safety_risk_level?: string | null;
  hitl_status?: string | null;
  ragas_scores?: Record<string, unknown>;
  source_count?: number | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type AskResponse = {
  query: string;
  answer: string;
  complexity: string;
  strategy: string;
  search_count: number;
  quality_score: number;
  session_id: string;
  request_id: string;
};

export type SourcesResponse = {
  sources: string[];
  total_chunks: number;
};

export type UploadResult = {
  filename: string;
  raw_segments: number;
  chunks: number;
  indexed: number;
  status: "ok" | "error";
  document_id?: string | null;
  source_document_id?: string | null;
  parse_quality_score?: number | null;
  outline_preview?: string | null;
  element_count?: number | null;
  warning_count?: number | null;
  uploaded_at?: string | null;
  chunk_strategy?: string | null;
  target_tokens?: number | null;
  overlap_tokens?: number | null;
  chunk_plan_reason?: string | null;
  error?: string | null;
};

export type UploadResponse = {
  request_id: string;
  results: UploadResult[];
  total_indexed: number;
};

export type SessionDocumentsResponse = {
  session_id: string;
  documents: UploadResult[];
  total_chunks: number;
};

export type SessionDocumentsDeleteResponse = {
  session_id: string;
  deleted: number;
};

export type EvalResponse = {
  query: string;
  direct_answer: Record<string, unknown>;
  standard_rag: Record<string, unknown>;
  adaptive_rag: Record<string, unknown>;
  conclusion: string;
  session_id?: string;
  request_id?: string;
};

export type DiagnosticsResponse = {
  langfuse: { enabled: boolean; configured: boolean; base_url: string };
  performance: Record<string, unknown>;
  tokens: Record<string, unknown>;
  service: Record<string, unknown>;
};
