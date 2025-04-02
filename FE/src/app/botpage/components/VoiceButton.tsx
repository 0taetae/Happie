"use client";

import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { sendMessage } from "../hooks/useChatbotResponse";

interface VoiceButtonProps {
  setQuestion: (q: string) => void;
  setAnswer: (a: string) => void;
  setStage: (s: "idle" | "recording" | "loading" | "answering") => void;
  label?: string; // 🔹 버튼 텍스트를 prop으로 받을 수 있게
}

export default function VoiceButton({
  setQuestion,
  setAnswer,
  setStage,
  label = "🎙 음성으로 질문하기", // 기본값
}: VoiceButtonProps) {
  const handleAudioComplete = (blob: Blob) => {
    setStage("loading");
    blob.arrayBuffer().then((buffer) => {
      const base64Data = Buffer.from(buffer).toString("base64");
      sendMessage(base64Data);
    });
  };

  const { startRecording } = useAudioRecorder(handleAudioComplete);

  const handleClick = () => {
    setQuestion("");
    setAnswer("");
    setStage("recording");
    startRecording();
  };

  return (
    <button
      onClick={handleClick}
      className="bg-gray-700 text-white px-5 py-3 rounded-full hover:bg-gray-800 shadow-lg"
    >
      {label}
    </button>
  );
}