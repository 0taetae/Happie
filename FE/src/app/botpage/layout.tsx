"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import HelpModal from "./components/HelpModal";
import BotRecorder from "./components/BotRecorder";
import BotChatBox from "./components/BotChatBox";
import { useAudioRecorder } from "./hooks/useAudioRecorder";
import { mqttClient } from "@/lib/mqttClient";

export default function BotLayout({ children }: { children: React.ReactNode }) {
  const [isListening, setIsListening] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);
  const [loading, setLoading] = useState(false);
  const [userText, setUserText] = useState("");       // 🙋 사용자 질문 텍스트
  const [responseText, setResponseText] = useState(""); // 🤖 응답 텍스트

  const handleAudioComplete = (blob: Blob) => {
    console.log("녹음 완료! Blob:", blob);
    setIsListening(false);
    setLoading(true);

    // 사용자의 질문 내용은 아직 없지만 표시용 텍스트 추가 가능
    setUserText("음성 메시지를 보냈어요 🎤");

    // MQTT로 오디오 전송
    blob.arrayBuffer().then((buffer) => {
      const base64Data = Buffer.from(buffer).toString("base64");
      mqttClient.publish("user/chatbot/request", base64Data);
      console.log("📤 MQTT 발신: user/chatbot/request");
    });
  };

  const { startRecording } = useAudioRecorder(handleAudioComplete);

  const handleAskClick = () => {
    setIsListening(true);
    setUserText("");       // 이전 내용 초기화
    setResponseText("");
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
      if (topic === "chatbot/response") {
        const text = message.toString();
        console.log("📩 서버 응답 수신:", text);
        setResponseText(text);
        setLoading(false);
        speakText(text);
      }
    });
  }, []);

  const speakText = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ko-KR";
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-100 px-4 py-6 relative">
      {/* 🤖 로봇 이미지 */}
      <div className="w-[25%] max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg">
        <Image
          src="/images/robot.jpg"
          alt="Chatbot"
          width={400}
          height={400}
          className="w-full h-auto rounded-full shadow-xl"
        />
      </div>

      {/* 🎤 음성 입력 버튼 or wave or 로딩 */}
      <BotRecorder
        isListening={isListening}
        loading={loading}
        onClick={handleAskClick}
      />

      {/* 💬 채팅 상자 */}
      <BotChatBox user={userText} bot={responseText} />

      {/* 📄 children 영역 */}
      <div className="mt-8 w-full max-w-4xl">{children}</div>

      {/* ❔ 도움말 버튼 */}
      <button
        onClick={() => setIsHelpOpen(true)}
        className="fixed bottom-6 right-6 w-16 h-16 bg-gray-400 text-white text-2xl font-bold rounded-full shadow-xl hover:bg-blue-500 transition"
        aria-label="도움말 열기"
      >
        ?
      </button>

      {isHelpOpen && (
        <HelpModal isOpen={isHelpOpen} fadeOut={fadeOut} onClose={handleClose} />
      )}
    </div>
  );
}
