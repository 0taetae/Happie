"use client";

import Image from "next/image";
import { useEffect, useRef } from "react";

interface WarningProps {
  onClose: () => void;
}

export default function Warning({ onClose }: WarningProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const audio = new Audio("/sounds/warning.mp3");
    audio.loop = true;

    const playAudio = async () => {
      try {
        await audio.play();
        console.log("🔊 경고음 재생 시작");
      } catch (err) {
        console.warn("🔇 경고음 재생 실패:", err);
      }
    };

    audioRef.current = audio;
    playAudio();

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        audioRef.current = null; // 🔁 명확히 초기화
      }
    };
  }, []);

  const handleClose = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null; // 🔁 명확히 초기화
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-60 px-4">
      <div className="bg-white border-4 border-red-500 w-full max-w-[90vw] md:max-w-3xl rounded-xl p-8 md:p-12 flex flex-col items-center animate-fadeInModal shadow-2xl">
        <div className="mb-6 w-full flex justify-center">
          <Image
            src="/images/fall-warning.png"
            alt="낙상 사고 이미지"
            width={600}
            height={400}
            className="object-contain w-full max-w-[85vw] md:max-w-[600px] h-auto aspect-[3/2] rounded-lg"
          />
        </div>

        <button
          onClick={handleClose}
          className="bg-red-500 text-white px-6 py-3 md:px-10 rounded-md hover:bg-red-600 transition font-semibold mt-4 w-full max-w-xs"
        >
          확인했습니다
        </button>
      </div>
    </div>
  );
}
