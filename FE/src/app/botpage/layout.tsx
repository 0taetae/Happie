"use client";

import Image from "next/image";
import { useState } from "react";
import HelpModal from "./components/HelpModal";
import { useAudioRecorder } from "./hooks/useAudioRecorder";

export default function BotLayout({ children }: { children: React.ReactNode }) {
  const [isListening, setIsListening] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);

  // 녹음 완료 시 처리
  const handleAudioComplete = (blob: Blob) => {
    console.log("녹음 완료! Blob:", blob);
    setIsListening(false);
    // MQTT 전송 처리 위치
  };

  const { startRecording } = useAudioRecorder(handleAudioComplete);

  const handleAskClick = () => {
    setIsListening(true); //버튼눌러서 녹음되는거 이미지로 보이기
    startRecording(); //5초녹음됨
  };

  const handleClose = () => {
    setFadeOut(true);
    setTimeout(() => {
      setIsHelpOpen(false);
      setFadeOut(false);
    }, 300);
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

      {/* 🎤 음성 녹음 버튼 또는 wave */}
      {!isListening ? (
        <div
          onClick={handleAskClick}
          role="button"
          className="mt-8 bg-blue-500 rounded-full shadow-lg px-8 py-4 cursor-pointer hover:bg-blue-600 transition flex items-center justify-center"
        >
          <span className="text-white text-lg font-semibold">하피에게 물어봐요!</span>
        </div>
      ) : (
        <div className="mt-8 w-[220px] h-[80px] bg-black rounded-[40px] shadow-lg flex items-center justify-center">
          <Image
            src="/images/voice-wave.gif"
            alt="Listening..."
            width={100}
            height={80}
            className="object-contain"
          />
        </div>
      )}

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

      {/* 🧾 모달 */}
      {isHelpOpen && (
        <HelpModal isOpen={isHelpOpen} fadeOut={fadeOut} onClose={handleClose} />
      )}
    </div>
  );
}
