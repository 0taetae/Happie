// hooks/useChatbotResponse.ts
import { useCallback } from "react";
import { mqttClient } from "@/lib/mqttClient";

export const sendMessage = (data: string) => {
  mqttClient.publish("user/chatbot/request", data);
  console.log("📤 MQTT 메시지 전송:", data);
};

type ChatbotResponseProps = {
  setQuestion: (q: string) => void;
  setAnswer: (a: string) => void;
  setStage: (s: "idle" | "recording" | "loading" | "answering") => void;
  setShowWarning: (show: boolean) => void;
  setFacility?: (f: string | null) => void;
};

export function useChatbotResponse({
  setQuestion,
  setAnswer,
  setStage,
  setShowWarning,
  setFacility,
}: ChatbotResponseProps) {
  const handleChatResponse = useCallback(
    (topic: string, message: Uint8Array | string) => {
      const msg = message.toString().trim();
      console.log("📩 수신된 메시지 원본:", msg);

      if (topic === "chatbot/response") {
        try {
          const parsed = JSON.parse(msg);
          setQuestion(parsed.request || "");
          setAnswer(parsed.response || msg);
          if (parsed.facility) setFacility?.(parsed.facility);
        } catch (e) {
          console.warn("❌ 파싱 실패:", e);
          setQuestion("");
          setAnswer(msg);
        }
        setStage("answering");
      }

      if (topic === "fall_detection") {
        setShowWarning(true);
      }
    },
    [setQuestion, setAnswer, setStage, setShowWarning, setFacility]
  );

  return { handleChatResponse };
}
