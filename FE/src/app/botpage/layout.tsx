"use client";

import { useEffect, useState, useCallback } from "react";
import Warning from "./components/Warning";
import QuestionButton from "./components/QuestionButton";
import VoiceButton from "./components/VoiceButton";
import EyeTracker from "./components/EyeTracker";
import { mqttClient } from "@/lib/mqttClient";
import { useChatbotResponse } from "./hooks/useChatbotResponse";
import Swal from "sweetalert2";
import "sweetalert2/dist/sweetalert2.min.css";

export type ColorType = "amber" | "rose" | "sky" | "lime" | "violet";

const colorOptions: ColorType[] = ["amber", "rose", "sky", "lime", "violet"];

const colorMap: Record<ColorType, string> = {
  amber: "bg-amber-100 border-amber-300 hover:bg-amber-200",
  sky: "bg-sky-100 border-sky-300 hover:bg-sky-200",
  rose: "bg-rose-100 border-rose-300 hover:bg-rose-200",
  lime: "bg-lime-100 border-lime-300 hover:bg-lime-200",
  violet: "bg-violet-100 border-violet-300 hover:bg-violet-200",
};

type Stage = "idle" | "recording" | "loading" | "answering";

export default function BotLayout() {
  const [stage, setStage] = useState<Stage>("idle");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [facility, setFacility] = useState<string | null>(null);
  const [showWarning, setShowWarning] = useState(false);
  const [selectedQuestion, setSelectedQuestion] = useState<string | null>(null);

  const questionList = [
    "원무수납처 어디야?",
    "심장혈관 조형실은 뭐하는 곳이야?",
    "501호실이 어디있어?",
  ].map((text, idx) => ({
    text,
    color: colorOptions[idx % colorOptions.length],
  }));

  const { handleChatResponse } = useChatbotResponse({
    setQuestion,
    setAnswer,
    setStage,
    setShowWarning,
    setFacility,
  });

  const onMqttMessage = useCallback(
    (topic: string, message: Buffer) => {
      handleChatResponse(topic, message);
    },
    [handleChatResponse]
  );

  useEffect(() => {
    mqttClient.on("message", onMqttMessage);
    return () => {
      mqttClient.removeListener("message", onMqttMessage);
    };
  }, [onMqttMessage]);

  useEffect(() => {
    if (stage === "answering" && answer) {
      const utter = new SpeechSynthesisUtterance(answer);
      utter.lang = "ko-KR";
      window.speechSynthesis.speak(utter);
    }
  }, [answer, stage]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 px-4 py-6 relative">
      <EyeTracker />

      {stage === "idle" && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-6 w-full max-w-4xl">
            {questionList.map(({ text, color }, idx) => (
              <QuestionButton
                key={idx}
                text={text}
                color={color}
                selected={selectedQuestion === text}
                onSelect={() => setSelectedQuestion(text)}
                setQuestion={setQuestion}
                setAnswer={setAnswer}
                setStage={setStage}
              />
            ))}
          </div>
          <VoiceButton setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} />
        </>
      )}

      {stage === "recording" && (
        <div className="text-center">
          <img src="/images/voice-wave.gif" alt="녹음 중" width={220} />
          <p className="mt-4">하피가 귀 기울이고 있어요...</p>
        </div>
      )}

      {stage === "loading" && (
        <div className="text-center">
          <img src="/images/voice-loading.gif" alt="로딩 중" width={100} />
          <p className="mt-2">하피가 대답을 준비 중이에요...</p>
        </div>
      )}

      {stage === "answering" && (
        <div className="w-full max-w-2xl relative">
          <div className="bg-white p-4 rounded-xl shadow mb-3">
            <p className="text-sm text-gray-500">🙋 질문</p>
            <p className="text-base">{question}</p>
          </div>
          <div className="bg-blue-50 p-4 rounded-xl shadow">
            <p className="text-sm text-blue-500">🤖 하피의 답변</p>
            <p className="text-base whitespace-pre-wrap">{answer || "하피가 응답 중이에요..."}</p>

            {answer?.endsWith("안내를 시작할까요?") && facility && (
              <div className="mt-4 flex justify-end gap-3">
                <button
                  onClick={async () => {
                    try {
                      const res = await fetch(`https://j12e103.p.ssafy.io/api/location/name`);
                      if (!res.ok) throw new Error("API 호출 실패");

                      Swal.fire({
                        icon: "success",
                        title: `${facility}로 안내를 시작합니다.`,
                        text: "로봇이 곧 출발할 예정이에요!",
                      });
                    } catch {
                      Swal.fire({
                        icon: "error",
                        title: "안내 시작 실패",
                        text: "죄송해요. 안내를 시작할 수 없어요 🥲",
                      });
                    }
                  }}
                  className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
                >
                  예
                </button>
                <button
                  onClick={() => setStage("idle")}
                  className="bg-gray-300 text-gray-800 px-4 py-2 rounded hover:bg-gray-400"
                >
                  아니요
                </button>
              </div>
            )}

            {!answer?.endsWith("안내를 시작할까요?") && (
              <div className="pt-5 flex justify-end gap-3">
                <button
                  onClick={() => setStage("idle")}
                  className="bg-blue-500 text-white px-4 py-2 rounded-md shadow hover:bg-blue-600 transition"
                >
                  홈으로 돌아가기
                </button>
                <VoiceButton
                  setQuestion={setQuestion}
                  setAnswer={setAnswer}
                  setStage={setStage}
                  label="🎤 음성으로 다시 질문하기"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {showWarning && <Warning onClose={() => setShowWarning(false)} />}
    </div>
  );
}
