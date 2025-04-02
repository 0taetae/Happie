"use client";

import { useEffect, useState, useCallback } from "react";
import Warning from "./components/Warning";
import QuestionButton from "./components/QuestionButton";
import VoiceButton from "./components/VoiceButton";
import { mqttClient } from "@/lib/mqttClient";
import { useChatbotResponse } from "./hooks/useChatbotResponse";
import Swal from "sweetalert2";
import "sweetalert2/dist/sweetalert2.min.css";

type Stage = "idle" | "recording" | "loading" | "answering";

export default function BotLayout() {
  const [stage, setStage] = useState<Stage>("idle");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [facility, setFacility] = useState<string | null>(null);
  const [showWarning, setShowWarning] = useState(false);

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
      {/* 질문 및 음성 입력 */}
      {stage === "idle" && (
        <>
          <div className="flex justify-center gap-4 mb-4 flex-wrap">
            {["원무수납처 어디야?", "심장혈관 조형실은 뭐하는 곳이야?", "502호실이 어디있어?"].map((text, idx) => (
              <QuestionButton key={idx} text={text} setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} />
            ))}
          </div>
          <VoiceButton setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} />
        </>
      )}

      {/* 녹음 중 */}
      {stage === "recording" && (
        <div className="text-center">
          <img src="/images/voice-wave.gif" alt="녹음 중" width={220} />
          <p className="mt-4">하피가 귀 기울이고 있어요...</p>
        </div>
      )}

      {/* 로딩 중 */}
      {stage === "loading" && (
        <div className="text-center">
          <img src="/images/voice-loading.gif" alt="로딩 중" width={100} />
          <p className="mt-2">하피가 대답을 준비 중이에요...</p>
        </div>
      )}

      {/* 답변 출력 */}
      {stage === "answering" && (
        <div className="w-full max-w-2xl relative">
          <div className="bg-white p-4 rounded-xl shadow mb-3">
            <p className="text-sm text-gray-500">🙋 질문</p>
            <p className="text-base">{question}</p>
          </div>
          <div className="bg-blue-50 p-4 rounded-xl shadow">
            <p className="text-sm text-blue-500">🤖 하피의 답변</p>
            <p className="text-base whitespace-pre-wrap">{answer || "하피가 응답 중이에요..."}</p>

            {/* 안내 유도 버튼 */}
            {answer?.endsWith("안내를 시작할까요?") && facility && (
              <div className="mt-4 flex justify-end gap-3">
                <button
                  onClick={async () => {
                    try {
                      const res = await fetch(`https://j12e103.p.ssafy.io/api/location/name/${facility}`);
                      if (!res.ok) throw new Error("API 호출 실패");

                      const data = await res.json();
                      console.log("✅ 안내 시작 API 응답:", data);

                      Swal.fire({
                        icon: "success",
                        title: `${facility}로 안내를 시작합니다.`,
                        text: "로봇이 곧 출발할 예정이에요!",
                        confirmButtonColor: "#3085d6",
                        confirmButtonText: "확인",
                      });

                      // ✅ 추후 stage 변경 등 로직 추가 가능
                    } catch (err) {
                      console.error("❌ 안내 API 오류:", err);
                      Swal.fire({
                        icon: "error",
                        title: "안내 시작 실패",
                        text: "죄송해요. 안내를 시작할 수 없어요 🥲",
                        confirmButtonText: "확인",
                      });
                    }
                  }}
                  className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
                >
                  예
                </button>

                <button onClick={() => setStage("idle")} className="bg-gray-300 text-gray-800 px-4 py-2 rounded hover:bg-gray-400">
                  아니요
                </button>
              </div>
            )}

            {/* ✅ 기본 종료 버튼: 응답이 완료된 경우에만 노출 */}
            {answer && !answer.endsWith("안내를 시작할까요?") && (
              <div className="pt-5 flex justify-end gap-3">
                <button onClick={() => setStage("idle")} className="bg-blue-500 text-white px-4 py-2 rounded-md shadow hover:bg-blue-600 transition">
                  홈으로 돌아가기
                </button>
                <VoiceButton setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} label="🎤 음성으로 다시 질문하기" />
              </div>
            )}
          </div>
        </div>
      )}

      {showWarning && <Warning onClose={() => setShowWarning(false)} />}
    </div>
  );
}
