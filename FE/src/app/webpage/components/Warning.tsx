"use client";

import Image from "next/image";
import { useEffect, useRef } from "react";

interface WarningProps {
  onClose: () => void;
  imageUrl: string;
}

export default function Warning({ onClose, imageUrl }: WarningProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      const playPromise = audio.play();

      if (playPromise !== undefined) {
        playPromise
          .then(() => console.log("🔊 경고음 재생됨"))
          .catch((err) => console.warn("🔇 경고음 재생 실패:", err));
      }
    }

    return () => {
      if (audio) {
        audio.pause();
        audio.currentTime = 0;
      }
    };
  }, []);

  const handleClose = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 px-4 py-10">
      {/* 🔊 오디오 자동 재생 */}
      <audio ref={audioRef} src="/sounds/warning.mp3" loop autoPlay />

      <div className="bg-white border-4 border-red-500 rounded-2xl p-6 sm:p-10 md:p-12 w-full max-w-6xl shadow-2xl animate-fadeInModal">
        {/* 제목 */}
        <div className="flex items-center justify-between mb-6">
          <span className="text-red-500 text-5xl">⚠️</span>
          <h2 className="text-2xl md:text-3xl font-bold text-center flex-grow">사고 발생</h2>
          <span className="text-red-500 text-5xl">⚠️</span>
        </div>

        {/* 이미지 영역 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="border-4 border-yellow-400 rounded-xl p-2 flex items-center justify-center">
            <Image
              src={imageUrl || "/images/fall.png"}
              alt="넘어진 노인"
              width={600}
              height={600}
              unoptimized
              className="object-contain w-full h-auto rounded-lg max-h-[50vh]"
            />
          </div>
          <div className="border-4 border-yellow-400 rounded-xl p-2 flex items-center justify-center">
            <Image
              src="/images/map.png"
              alt="사고 위치"
              width={600}
              height={600}
              className="object-contain w-full h-auto rounded-lg max-h-[50vh]"
            />
          </div>
        </div>

        {/* 버튼 */}
        <div className="flex justify-center">
          <button
            onClick={handleClose}
            className="bg-red-500 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-red-600 transition"
          >
            확인했습니다
          </button>
        </div>
      </div>
    </div>
  );
}
