import { create } from "zustand";
import type { ChatMessage, SourceDoc, TraceSummary } from "@/lib/types";

type ChatState = {
  messages: ChatMessage[];
  trace: TraceSummary;
  sources: SourceDoc[];
  setMessages: (messages: ChatMessage[]) => void;
  setTrace: (trace: TraceSummary) => void;
  setSources: (sources: SourceDoc[]) => void;
  clear: () => void;
};

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  trace: {},
  sources: [],
  setMessages: (messages) => set({ messages }),
  setTrace: (trace) => set({ trace }),
  setSources: (sources) => set({ sources }),
  clear: () => set({ messages: [], trace: {}, sources: [] })
}));
