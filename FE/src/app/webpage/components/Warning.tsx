"use client";

import { useState } from "react";
import Image from "next/image";

export default function Warning() {
  const [show, setShow] = useState(false);

  return (
    <>
      {/* 버튼 */}
      <button
        onClick={() => setShow(true)}
        className="bg-red-500 text-white px-4 py-2 rounded-lg shadow hover:bg-red-600"
      >
        ⚠️ 사고 알림
      </button>

      {/* 모달 */}
      {show && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div
            className="bg-white border-2 border-red-300 rounded-xl p-6 w-[600px] relative shadow-lg animate-fadeInModal"
          >
            {/* 닫기 버튼 */}
            <button
              onClick={() => setShow(false)}
              className="absolute top-3 right-3 text-2xl font-bold text-gray-600 hover:text-black"
            >
              X
            </button>

            {/* 경고 아이콘 + 제목 */}
            <div className="flex items-center justify-between mb-4">
              <span className="text-red-500 text-4xl">⚠️</span>
              <h2 className="text-xl font-bold text-center flex-grow">사고 발생</h2>
              <span className="text-red-500 text-4xl">⚠️</span>
            </div>

            {/* 이미지 영역 */}
            <div className="grid grid-cols-2 gap-4 mt-2">
              <div className="border-4 border-yellow-400 rounded p-1 flex items-center justify-center">
                <Image
                  src="/images/fall.png"
                  alt="넘어진 노인"
                  width={220}
                  height={220}
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
          </div>
        </div>
      )}
    </>
  );
}
