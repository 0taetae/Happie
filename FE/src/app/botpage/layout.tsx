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
import { colorOptions } from "../../types/color";
import "./../globals.css";

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

  const questionList = ["원무수납처 \n어디야?", "소아진정실은 \n뭐하는 곳이야?", "501호실이 \n어디있어?"].map((text, idx) => ({
    text,
    color: colorOptions[idx % colorOptions.length],
  }));

  const { handleChatResponse } = useChatbotResponse({
    setQuestion,
    setAnswer,
    setStage,
    setShowWarning,
    setFacility,
    setNavigationImage, // ✅ 수신된 이미지 기억해두기
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

      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(answer);
      utter.lang = "ko-KR";
      window.speechSynthesis.speak(utter);

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
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 relative">
      {stage !== "navigating" && <EyeTracker />}

      {/* ✅ 질문 버튼 + 음성 질문 */}
      {stage === "idle" && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-6 w-full max-w-4xl">
            {questionList.map(({ text, color }, idx) => (
              <QuestionButton key={idx} text={text} color={color} selected={selectedQuestion === text} onSelect={() => setSelectedQuestion(text)} setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} />
            ))}
          </div>
          <VoiceButton setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} stage={stage} size={24} />
        </>
      )}

      {/* ✅ 녹음 중 UI */}
      {stage === "recording" && (
        <>
          <div className="absolute top-[calc(50%-7.5rem)] left-[calc(50%+20rem)] z-20 animate-slideInFromRight">
            <img src="/images/ear.png" alt="귀 기울이는 중" className="w-[150px] h-[150px] animate-scalePulse" />
          </div>
          <div className="flex flex-col items-center justify-center text-center bg-white w-[400px] h-[100px] p-4 rounded-xl shadow-md">
            <p className="text-2xl text-gray-700 flex items-center">
              하피가 귀 기울이고 있어요 <DotAnimation />
            </p>
          </div>
        </>
      )}

      {/* ✅ 응답 로딩 중 UI */}
      {stage === "loading" && (
        <div className="flex items-center bg-white w-[400px] h-[100px] p-4 rounded-xl shadow-md">
          <img src="/images/voice-loading1.gif" alt="로딩 중" className="w-16 h-16 mr-4 ml-6" />
          <p className="text-2xl text-gray-800 flex items-center">
            흠.. 그게 말이죠 <DotAnimation />
          </p>
        </div>
      )}

      {/* ✅ 답변 출력 UI */}
      {stage === "answering" && (
        <div className="w-full max-w-5xl relative">
          <div className="bg-white p-6 rounded-xl shadow mb-3">
            <p className="text-2xl text-gray-500 mb-4">❓ 질문</p>
            <p className="text-3xl">{question}</p>
          </div>
          <div className="bg-blue-50 p-6 rounded-xl shadow">
            <p className="text-2xl text-blue-500 mb-4">🤖 하피의 답변</p>
            <p className="text-3xl whitespace-pre-wrap">
              {typedAnswer || (
                <div className="text-3xl wavy-text flex gap-0.5">
                  {"하피가 응답 중이에요".split("").map((char, idx) => (
                    <span key={idx}>{char}</span>
                  ))}
                  <DotAnimation />
                </div>
              )}
            </p>

            {/* ✅ 안내 시작 여부 확인 */}
            {isTypingDone && answer?.endsWith("안내를 시작할까요?") && facility && (
              <div className="flex gap-5 mt-4 justify-end items-center">
                <button
                  onClick={async () => {
                    window.speechSynthesis.cancel();
                    const result = await Swal.fire({
                      title: "안내를 시작할까요?",
                      text: `${facility}로 안내를 시작합니다.`,
                      icon: "question",
                      showCancelButton: true,
                      confirmButtonText: "네, 시작할게요!",
                      cancelButtonText: "취소",
                      reverseButtons: true,
                    });

                    if (result.isConfirmed) {
                      setStage("navigating"); // ✅ 이전에 받은 image 사용
                    }
                  }}
                  className="w-20 h-10 bg-green-200 hover:bg-green-300 rounded-xl text-gray-800 shadow-md transition-all"
                >
                  예
                </button>
                <button
                  onClick={() => {
                    window.speechSynthesis.cancel();
                    setStage("idle");
                    setAnswer("");
                    setTypedAnswer("");
                    setIsTypingDone(false);
                  }}
                  className="w-20 h-10 bg-red-200 hover:bg-red-300 rounded-xl text-gray-800 shadow-md transition-all"
                >
                  아니오
                </button>
              </div>
            )}

            {/* ✅ 다시 질문하기 */}
            {isTypingDone && !answer?.endsWith("안내를 시작할까요?") && (
              <div className="pt-3 flex justify-end items-center gap-6">
                <VoiceButton setQuestion={setQuestion} setAnswer={setAnswer} setStage={setStage} stage={stage} size={14} />
                <button
                  onClick={() => {
                    window.speechSynthesis.cancel();
                    setStage("idle");
                    setAnswer("");
                    setTypedAnswer("");
                    setIsTypingDone(false);
                  }}
                  className="w-20 h-10 bg-indigo-500 text-white hover:bg-indigo-700 rounded-xl shadow-md transition-all"
                >
                  홈으로
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ✅ 안내 중 화면 */}
      {stage === "navigating" && navigationImage && (
        <div className="flex flex-col items-center justify-center bg-white w-full h-full flex-grow gap-10">
          <img src={navigationImage} alt="안내 중" className="rounded-xl w-full max-h-[600px] object-contain mb-4" />
          <p className="text-xl text-gray-800 font-semibold flex items-center">
            <div className="text-7xl text-emerald-700 wavy-text flex gap-2">
              {"저만 믿고 따라오세요!".split("").map((char, idx) => (
                <span key={idx}>{char}</span>
              ))}
            </div>
          </p>
        </div>
      )}

      {showWarning && <Warning onClose={() => setShowWarning(false)} />}
    </div>
  );
}
