"use client";

import Image from "next/image";

interface WarningProps {
  onClose: () => void;
}

export default function Warning({ onClose }: WarningProps) {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-60">
      <div className="bg-white w-full h-full flex flex-col items-center justify-center relative animate-fadeInModal">
        <div className="flex justify-center mb-6">
          <Image
            src="/images/fall-warning.png"
            alt="낙상 사고 이미지"
            width={750}
            height={750}
            className="object-contain"
          />
        </div>

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
