"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [inputCode, setInputCode] = useState(""); // 코드 입력 상태
  const router = useRouter();

  return (
    <div className="flex h-screen">
      {/* 왼쪽 화면 (WEB) */}
      <div 
        className="w-1/2 flex items-center justify-center bg-blue-300 text-white text-2xl font-bold cursor-pointer hover:bg-blue-400 transition"
        onClick={() => setIsLoginOpen(true)}
      >
        WEB
      </div>

      {/* 오른쪽 화면 (BOT) */}
      <div 
        className="w-1/2 flex items-center justify-center bg-green-400 text-white text-2xl font-bold cursor-pointer hover:bg-green-500 transition"
        onClick={() => router.push("/botpage/main")}
      >
        BOT
      </div>

      {/* 로그인 모달 */}
      {isLoginOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-80 text-center">
            <h2 className="text-lg font-bold mb-4">로그인</h2>
            <input
              type="text"
              placeholder="코드를 입력하세요"
              value={inputCode}
              onChange={(e) => setInputCode(e.target.value)}
              className="w-full p-2 border rounded mb-4"
            />
            <div className="flex justify-end space-x-2">
              <button
                className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500 transition"
                onClick={() => setIsLoginOpen(false)}
              >
                취소
              </button>
              <button
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700 transition"
                onClick={() => router.push("/webpage/home")}
              >
                확인
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
