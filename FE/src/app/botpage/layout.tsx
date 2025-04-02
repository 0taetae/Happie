"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import HelpModal from "./components/HelpModal";
import Warning from "./components/Warning";
import { useAudioRecorder } from "./hooks/useAudioRecorder";
import { mqttClient } from "@/lib/mqttClient";

type Stage = "idle" | "recording" | "loading" | "answering";

export default function BotLayout({ children }: { children: React.ReactNode }) {
  const [stage, setStage] = useState<Stage>("idle");
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [displayedAnswer, setDisplayedAnswer] = useState("");

  // ⛑️ 낙상 감지 상태
  const [showWarning, setShowWarning] = useState(false);

  const handleAudioComplete = (blob: Blob) => {
    setStage("loading");

    blob.arrayBuffer().then((buffer) => {
      const base64Data = Buffer.from(buffer).toString("base64");
      mqttClient.publish("user/chatbot/request", base64Data);
      console.log("📤 MQTT 발신 완료");
    });
  };

  const { startRecording } = useAudioRecorder(handleAudioComplete);

  const handleAskClick = () => {
    setQuestion("");
    setAnswer("");
    setDisplayedAnswer("");
    setStage("recording");
    startRecording();
  };

  const handleClose = () => {
    setFadeOut(true);
    setTimeout(() => {
      setIsHelpOpen(false);
      setFadeOut(false);
    }, 300);
  };

  useEffect(() => {
    mqttClient.on("message", (topic, message) => {
      const msg = message.toString();

      // 🤖 챗봇 응답 수신
      if (topic === "chatbot/response") {
        try {
          const { quest, answer } = JSON.parse(msg);
          setQuestion(quest);
          setAnswer(answer);
          setDisplayedAnswer("");
          setStage("answering");
        } catch (e) {
          console.error("❌ 응답 파싱 실패:", e);
        }
      }

      // ⛑️ 낙상 감지 수신
      if (topic === "fall_detection") {
        try {
          setShowWarning(true);
        } catch (e) {
          console.error("❌ 낙상 데이터 파싱 실패:", e);
        }
      }
    });
  }, []);

  // 💬 답변 타이핑 효과 + TTS
  useEffect(() => {
    if (stage !== "answering" || !answer) return;

    let i = 0;
    const interval = setInterval(() => {
      setDisplayedAnswer((prev) => {
        const nextChar = answer[i];
        i++;
        if (i >= answer.length) clearInterval(interval);
        return prev + (nextChar ?? "");
      });
    }, 60);

    speakText(answer);
    return () => clearInterval(interval);
  }, [answer, stage]);

  const speakText = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ko-KR";
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 px-4 py-6 relative">
      {/* 단계별 화면 렌더링 */}
      {stage === "idle" && (
        <div className="flex flex-col items-center justify-center text-center">
          <Image src="/images/robot.jpg" alt="Chatbot" width={300} height={300} className="rounded-full shadow-xl mb-6" />
          <button onClick={handleAskClick} className="bg-blue-500 text-white px-6 py-3 rounded-full shadow-lg hover:bg-blue-600 transition">
            하피에게 물어봐요!
          </button>
        </div>
      )}

      {stage === "recording" && (
        <div className="flex flex-col items-center justify-center text-center">
          <Image src="/images/voice-wave.gif" alt="녹음 중" width={220} height={80} className="object-contain" />
          <span className="mt-4 text-gray-600 font-medium">하피가 귀 기울이고 있어요...</span>
        </div>
      )}

      {stage === "loading" && (
        <div className="flex flex-col items-center justify-center text-center">
          <Image src="/images/voice-loading.gif" alt="로딩 중" width={100} height={100} className="object-contain" />
          <span className="mt-2 text-gray-600 font-medium">하피가 대답을 준비 중이에요...</span>
        </div>
      )}

      {stage === "answering" && (
        <div className="w-full max-w-2xl">
          <div className="bg-white p-4 rounded-xl shadow mb-3 text-left">
            <p className="text-sm text-gray-500 mb-1">🙋‍♀️ 질문</p>
            <div className="text-base text-gray-800">{question}</div>
          </div>
          <div className="bg-blue-50 p-4 rounded-xl shadow text-left">
            <p className="text-sm text-blue-500 mb-1">🤖 하피의 답변</p>
            <div className="text-base text-gray-800 whitespace-pre-wrap">{displayedAnswer}</div>
          </div>
        </div>
      )}

      {/* children */}
      <div className="mt-8 w-full max-w-4xl">{children}</div>

      {/* 도움말 버튼 */}
      <button onClick={() => setIsHelpOpen(true)} className="fixed bottom-6 right-6 w-16 h-16 bg-gray-400 text-white text-2xl font-bold rounded-full shadow-xl hover:bg-blue-500 transition" aria-label="도움말 열기">
        ?
      </button>

      {isHelpOpen && <HelpModal isOpen={isHelpOpen} fadeOut={fadeOut} onClose={handleClose} />}

      {/* 낙상 경고 모달 */}
      {showWarning && <Warning onClose={() => setShowWarning(false)} />}
    </div>
  );
}
