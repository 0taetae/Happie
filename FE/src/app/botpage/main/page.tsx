"use client";

import Image from "next/image";

export default function BotMain() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 relative">
      {/* 로봇 이미지 */}
      <Image
        src="/images/robot.jpg" // 로봇 이미지 경로
        alt="Chatbot"
        width={200}
        height={200}
        className="rounded-full shadow-lg"
      />

      {/* 말풍선 버튼 */}
      <button className="mt-6 px-6 py-3 bg-blue-400 text-white text-lg font-bold rounded-full shadow-lg hover:bg-blue-500 transition">
        하피에게 물어봐요!
      </button>

      {/* 도움말 버튼 (우측 하단) */}
      <button className="absolute bottom-6 right-6 bg-blue-200 text-blue-600 p-3 rounded-full shadow-lg hover:bg-blue-300 transition">
        ?
      </button>
    </div>
  );
}
