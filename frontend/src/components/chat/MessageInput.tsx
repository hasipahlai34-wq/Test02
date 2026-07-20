"use client";

import { FormEvent, KeyboardEvent, useState } from "react";
import { Send, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

type MessageInputProps = {
  disabled?: boolean;
  generating?: boolean;
  onStop?: () => void;
  onSubmit: (query: string) => void;
};

export function MessageInput({ disabled, generating, onStop, onSubmit }: MessageInputProps) {
  const [value, setValue] = useState("");

  function submitValue() {
    const query = value.trim();
    if (!query || disabled) return;
    setValue("");
    onSubmit(query);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    submitValue();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    const nativeEvent = event.nativeEvent as KeyboardEvent<HTMLTextAreaElement>["nativeEvent"] & { isComposing?: boolean };
    if (event.key !== "Enter" || event.shiftKey || nativeEvent.isComposing) {
      return;
    }
    event.preventDefault();
    submitValue();
  }

  return (
    <form onSubmit={submit} className="flex gap-3">
      <Textarea disabled={disabled} value={value} onChange={(event) => setValue(event.target.value)} onKeyDown={handleKeyDown} placeholder="输入你的问题，例如：这个项目的 Adaptive RAG 流程是什么？" className="min-h-20 resize-none" />
      {generating && onStop ? (
        <Button type="button" variant="outline" onClick={onStop} className="self-end gap-2"><Square className="h-4 w-4" />停止</Button>
      ) : (
        <Button type="submit" disabled={disabled || !value.trim()} className="self-end gap-2"><Send className="h-4 w-4" />发送</Button>
      )}
    </form>
  );
}
