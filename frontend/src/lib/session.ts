const SESSION_KEY = "adaptive-rag.session_id";

export function getOrCreateSessionId() {
  if (typeof window === "undefined") return `browser-${crypto.randomUUID()}`;
  const existing = window.localStorage.getItem(SESSION_KEY);
  if (existing) return existing;
  const created = `browser-${crypto.randomUUID()}`;
  window.localStorage.setItem(SESSION_KEY, created);
  return created;
}

export function resetSessionId() {
  const created = `browser-${crypto.randomUUID()}`;
  window.localStorage.setItem(SESSION_KEY, created);
  return created;
}

export function createRequestId() {
  return crypto.randomUUID();
}
