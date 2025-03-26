"use client";

import Image from "next/image";
import { useState, useEffect } from "react";
import HelpModal from "./components/HelpModal"; // 필요 시 분리된 모달 컴포넌트

export default function BotLayout({ children }: { children: React.ReactNode }) {
  const [isListening, setIsListening] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);

  // 5초 후 음성 인식 종료
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isListening) {
      timer = setTimeout(() => setIsListening(false), 5000);
    }
    return () => clearTimeout(timer);
  }, [isListening]);

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

      {/* 🎤 말풍선 or 음성 파형 */}
      {!isListening ? (
        <div
          onClick={() => setIsListening(true)}
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
