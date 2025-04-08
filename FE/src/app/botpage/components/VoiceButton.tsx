"use client";

import { useRef } from "react";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { sendMessage } from "../hooks/useChatbotResponse";

interface VoiceButtonProps {
  setQuestion: (q: string) => void;
  setAnswer: (a: string) => void;
  setStage: (s: "idle" | "recording" | "loading" | "answering") => void;
  size?: number;
}

export default function VoiceButton({
  setQuestion,
  setAnswer,
  setStage,
  size = 14, // 기본 사이즈
}: VoiceButtonProps) {
  const buttonRef = useRef<HTMLButtonElement | null>(null);

  const { startRecording } = useAudioRecorder((blob: Blob) => {
    setStage("loading");
    blob.arrayBuffer().then((buffer) => {
      const base64Data = Buffer.from(buffer).toString("base64");
      sendMessage(base64Data);
    });
  });

  const handleClick = () => {
    if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
      window.speechSynthesis.cancel(); // ✅ TTS 중단
    }

    setQuestion("");
    setAnswer("");
    setStage("recording");
    startRecording();
  };

  return (
    <button
      ref={buttonRef}
      onClick={handleClick}
      className={`w-${size} h-${size} rounded-full flex items-center justify-center hover:scale-105 transition`}
    >
      <img src="/images/mic.png" alt="음성 질문" className={`w-${size} h-${size}`} />
    </button>
  );
}



