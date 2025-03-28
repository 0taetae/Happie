"use client";

import Image from "next/image";

interface WarningProps {
  onClose: () => void;
  imageUrl: string; // 🔹 추가된 이미지 URL prop
}

export default function Warning({ onClose, imageUrl }: WarningProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div className="bg-white border-2 border-red-300 rounded-xl p-6 w-[600px] relative shadow-lg animate-fadeInModal">
        {/* 경고 제목 */}
        <div className="flex items-center justify-between mb-4">
          <span className="text-red-500 text-4xl">⚠️</span>
          <h2 className="text-xl font-bold text-center flex-grow">사고 발생</h2>
          <span className="text-red-500 text-4xl">⚠️</span>
        </div>

        {/* 이미지 */}
        <div className="grid grid-cols-2 gap-4 mt-2 mb-6">
          <div className="border-4 border-yellow-400 rounded p-1 flex items-center justify-center">
            <Image
              src={imageUrl || "/images/fall.png"} // 🔸 imageUrl 사용, fallback도 제공
              alt="넘어진 노인"
              width={220}
              height={220}
              unoptimized // S3 같은 외부 이미지일 경우 필수
            />
          </div>
          <div className="border-4 border-yellow-400 rounded p-1 flex items-center justify-center">
            <Image
              src="/images/map.png"
              alt="사고 위치"
              width={220}
              height={220}
            />
          </div>
        </div>

        {/* 중앙 확인 버튼 */}
        <div className="flex justify-center">
          <button
            onClick={onClose}
            className="bg-red-500 text-white px-6 py-2 rounded-md hover:bg-red-600 transition font-semibold"
          >
            확인했습니다
          </button>
        </div>
      </div>
    </div>
  );
}
