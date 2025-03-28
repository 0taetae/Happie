"use client";

import Image from "next/image";

interface BotRecorderProps {
  isListening: boolean;
  loading: boolean;
  onClick: () => void;
}

export default function BotRecorder({ isListening, loading, onClick }: BotRecorderProps) {
  if (loading) {
    return (
      <div className="mt-8 flex flex-col items-center justify-center">
        <Image
          src="/images/voice-loading.gif"
          alt="로딩 중"
          width={100}
          height={100}
          className="object-contain"
        />
        <span className="mt-2 text-gray-600 font-medium">하피가 대답을 준비하는 중...</span>
      </div>
    );
  }

  if (isListening) {
    return (
      <div className="mt-8 w-[220px] h-[80px] bg-black rounded-[40px] shadow-lg flex items-center justify-center">
        <Image
          src="/images/voice-wave.gif"
          alt="Listening..."
          width={100}
          height={80}
          className="object-contain"
        />
      </div>
    );
  }

  return (
    <div
      onClick={onClick}
      role="button"
      className="mt-8 bg-blue-500 rounded-full shadow-lg px-8 py-4 cursor-pointer hover:bg-blue-600 transition flex items-center justify-center"
    >
      <span className="text-white text-lg font-semibold">하피에게 물어봐요!</span>
    </div>
  );
}
