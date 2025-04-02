"use client";

import { useCallback } from "react";

type ChatbotResponseProps = {
  setQuestion: (q: string) => void;
  setAnswer: (a: string) => void;
  setStage: (s: "idle" | "recording" | "loading" | "answering") => void;
  setShowWarning: (show: boolean) => void;
};

const useChatbotResponse = ({
  setQuestion,
  setAnswer,
  setStage,
  setShowWarning,
}: ChatbotResponseProps) => {
  const handleChatResponse = useCallback(
    (topic: string, message: Buffer) => {
      const msg = message.toString().trim();
      console.log("📩 수신된 메시지 원본:", JSON.stringify(msg));

      if (topic === "chatbot/response") {
        try {
          const parsed = JSON.parse(msg);
          if (parsed.request && parsed.response) {
            setQuestion(parsed.request);
            setAnswer(parsed.response);
          } else {
            setQuestion("");
            setAnswer(msg);
          }
        } catch (e) {
          console.error("❌ 응답 파싱 실패:", e);
          setQuestion("");
          setAnswer(msg);
        }
        setStage("answering");
      }

      if (topic === "fall_detection") {
        setShowWarning(true);
      }
    },
    [setQuestion, setAnswer, setStage, setShowWarning]
  );

  return { handleChatResponse };
};

export default useChatbotResponse;
