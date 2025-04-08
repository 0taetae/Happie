"use client";

import { useRef } from "react";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { sendMessage } from "../hooks/useChatbotResponse";

interface VoiceButtonProps {
  setQuestion: (q: string) => void;
  setAnswer: (a: string) => void;
  setStage: (s: "idle" | "recording" | "loading" | "answering") => void;
  stage: "idle" | "recording" | "loading" | "answering";
  size?: number;
}

export default function VoiceButton({
  setQuestion,
  setAnswer,
  setStage,
  stage,
  size = 14,
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
      window.speechSynthesis.cancel();
    }

    setQuestion("");
    setAnswer("");
    setStage("recording");
    startRecording();
  };

  return (
    <>
      {stage === "idle" ? (
        <button
          ref={buttonRef}
          onClick={handleClick}
          className={`w-${size} h-${size} rounded-full flex items-center justify-center hover:scale-105 transition`}
        >
          <img
            src="/images/mic.png"
            alt="음성 질문"
            className={`w-${size} h-${size}`}
          />
        </button>
      ) : (
        <button
          ref={buttonRef}
          onClick={handleClick}
          className="w-32 h-10 bg-emerald-500 text-white border border-gray-300 hover:bg-emerald-700 rounded-xl text-gray-800 shadow-md hover:scale-110 transition-all duration-200 text-lg"
        >
          다시 질문하기
        </button>
      )}
    </>
  );
}
