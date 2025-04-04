"use client";

import { useEffect, useState, useCallback } from "react";
import Warning from "./components/Warning";
import QuestionButton from "./components/QuestionButton";
import VoiceButton from "./components/VoiceButton";
import DotAnimation from "./components/DotAnimation";
import EyeTracker from "./components/EyeTracker";
import { mqttClient } from "@/lib/mqttClient";
import { useChatbotResponse } from "./hooks/useChatbotResponse";
import Swal from "sweetalert2";
import "sweetalert2/dist/sweetalert2.min.css";
import { colorOptions } from "../../types/color";

type Stage = "idle" | "recording" | "loading" | "answering" | "navigating";

export default function BotLayout() {
  const [stage, setStage] = useState<Stage>("idle");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [typedAnswer, setTypedAnswer] = useState("");
  const [isTypingDone, setIsTypingDone] = useState(false);
  const [facility, setFacility] = useState<string | null>(null);
  const [showWarning, setShowWarning] = useState(false);
  const [selectedQuestion, setSelectedQuestion] = useState<string | null>(null);
  const [navigationImage, setNavigationImage] = useState<string | null>(null);

  const questionList = ["원무수납처 어디야?", "소아진정실은 뭐하는 곳이야?", "501호실이 어디있어?"].map((text, idx) => ({
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
    if (stage === "loading" || stage === "recording") {
      setAnswer("");
      setTypedAnswer("");
      setIsTypingDone(false);
    }
  }, [stage]);

  useEffect(() => {
    if (stage === "answering" && answer) {
      setTypedAnswer("");
      setIsTypingDone(false);

      // ✅ TTS 먼저 실행
      window.speechSynthesis.cancel(); // 혹시 전에 재생 중이면 중단
      const utter = new SpeechSynthesisUtterance(answer);
      utter.lang = "ko-KR";
      window.speechSynthesis.speak(utter);

      // ✅ 타자 효과 시작
      const steps = Array.from({ length: answer.length }, (_, i) => answer.slice(0, i + 1));
      let idx = 0;

      const interval = setInterval(() => {
        setTypedAnswer(steps[idx]);
        idx++;

        if (idx >= steps.length) {
          clearInterval(interval);
          setIsTypingDone(true);
        }
      }, 150);

      return () => clearInterval(interval);
    }
  }, [answer, stage]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 px-4 py-6 relative">
      <EyeTracker />

      {stage === "idle" && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-6 w-full max-w-4xl">
            {questionList.map(({ text, color }, idx) => (
              <QuestionButton key={idx} text={text} color={color} selected={selectedQuestion === text} onSelect={() => setSelectedQuestion(text)} setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} />
            ))}
          </div>
          <VoiceButton setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} size={24} />
        </>
      )}

      {stage === "recording" && (
        <>
          <div className="absolute top-[calc(50%-7.5rem)] left-[calc(50%+20rem)] z-20 animate-slideInFromRight">
            <img src="/images/ear.png" alt="귀 기울이는 중" className="w-[150px] h-[150px] animate-scalePulse" />
          </div>

          <div className="flex flex-col items-center justify-center text-center bg-white w-[400px] h-[100px] p-4 rounded-xl shadow-md">
            <p className="text-2xl font-semibold text-gray-700 flex items-center">
              하피가 귀 기울이고 있어요
              <span className="ml-1">
                <DotAnimation />
              </span>
            </p>
          </div>
        </>
      )}

      {stage === "loading" && (
        <div className="flex items-center bg-white w-[400px] h-[100px] p-4 rounded-xl shadow-md">
          {/* 👈 이미지 왼쪽 고정 */}
          <img src="/images/voice-loading1.gif" alt="로딩 중" className="w-16 h-16 mr-4 ml-6" />

          {/* 👉 텍스트 우측에 표시 */}
          <p className="text-2xl font-semibold text-gray-800 flex items-center">
            흠.. 그게 말이죠
            <span className="ml-1">
              <DotAnimation />
            </span>
          </p>
        </div>
      )}

      {stage === "answering" && (
        <div className="w-full max-w-3xl relative">
          <div className="bg-white p-6 rounded-xl shadow mb-3">
            <p className="text-xl font-semibold text-gray-500 mb-2">❓ 질문</p>
            <p className="text-lg">{question}</p>
          </div>
          <div className="bg-blue-50 p-6 rounded-xl shadow">
            <p className="text-xl font-semibold text-blue-500 mb-2">🤖 하피의 답변</p>
            <p className="text-lg whitespace-pre-wrap">
              {typedAnswer || (
                <>
                  하피가 응답 중이에요
                  <DotAnimation />
                </>
              )}
            </p>

            {isTypingDone && answer?.endsWith("안내를 시작할까요?") && facility && (
              <div className="mt-4 flex justify-end gap-6 items-center">
                {/* ✅ 안내 시작 (예) */}
                <img
                  src="/images/yes.png"
                  alt="안내 시작"
                  title="안내 시작"
                  onClick={async () => {
                    window.speechSynthesis.cancel(); // TTS 중단
                    try {
                      const res = await fetch(`https://j12e103.p.ssafy.io/api/location/name`, {
                        method: "POST",
                        headers: {
                          "Content-Type": "application/json",
                        },
                        body: JSON.stringify({ name: facility }),
                      });
                      if (!res.ok) throw new Error("API 호출 실패");

                      const data = await res.json();

                      setNavigationImage(data.image); // ✅ 이미지 저장
                      setStage("navigating"); // ✅ 새로운 단계로 전환
                    } catch {
                      Swal.fire({
                        icon: "error",
                        title: "안내 시작 실패",
                        text: "죄송해요. 안내를 시작할 수 없어요 🥲",
                      });
                    }
                  }}
                  className="w-14 h-14 cursor-pointer hover:scale-110 transition"
                />

                {/* ✅ 홈으로 이동 (아니요) */}
                <img
                  src="/images/no.png"
                  alt="홈으로"
                  title="홈으로 돌아가기"
                  onClick={() => {
                    window.speechSynthesis.cancel();
                    setStage("idle");
                    setAnswer("");
                    setTypedAnswer("");
                    setIsTypingDone(false);
                  }}
                  className="w-14 h-14 cursor-pointer hover:scale-110 transition"
                />
              </div>
            )}

            {isTypingDone && !answer?.endsWith("안내를 시작할까요?") && (
              <div className="pt-2 flex justify-end gap-4">
                <VoiceButton setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} size={14} />
                <img
                  src="/images/home.png"
                  alt="홈으로"
                  onClick={() => {
                    window.speechSynthesis.cancel();
                    setStage("idle");
                    setAnswer("");
                    setTypedAnswer("");
                    setIsTypingDone(false);
                  }}
                  className="w-14 h-14 cursor-pointer-custom hover:scale-105 transition"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {stage === "navigating" && navigationImage && (
        <div className="flex flex-col items-center justify-center bg-white p-6 rounded-xl shadow w-full max-w-xl">
          <img src={navigationImage} alt="안내 중" className="rounded-xl w-full max-h-[400px] object-contain mb-4" />
          <p className="text-xl font-semibold text-gray-800 flex items-center">
            하피와 함께 이동 중입니다
            <DotAnimation />
          </p>
        </div>
      )}

      {showWarning && <Warning onClose={() => setShowWarning(false)} />}
    </div>
  );
}
