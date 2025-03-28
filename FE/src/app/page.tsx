"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Swal from "sweetalert2";

export default function Home() {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [inputCode, setInputCode] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // 홈 진입 시 access_code 초기화
  useEffect(() => {
    localStorage.removeItem("access_code");
  }, []);

  // 모달 열릴 때 자동 포커스
  useEffect(() => {
    if (isLoginOpen) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100); // 약간의 딜레이 후 안정적으로 포커스
    }
  }, [isLoginOpen]);

  const handleLogin = () => {
    if (inputCode === "103") {
      Swal.fire({
        icon: "success",
        title: "로그인 성공!",
        text: "Happie에 오신 걸 환영합니다 😊",
        showConfirmButton: false,
        timer: 1500,
      });

      setTimeout(() => {
        localStorage.setItem("access_code", "103");
        router.push("/webpage/home");
      }, 1500);
    } else {
      Swal.fire({
        icon: "error",
        title: "인증 실패",
        text: "코드가 틀렸습니다. 다시 입력해주세요!",
        confirmButtonColor: "#3085d6",
        confirmButtonText: "확인",
      });
    }
  };

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
        onClick={() => router.push("/botpage")}
      >
        BOT
      </div>

      {/* 로그인 모달 */}
      {isLoginOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/50 z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-80 text-center">
            <h2 className="text-lg font-bold mb-4">로그인</h2>

            <input
              ref={inputRef}
              type="text"
              placeholder="코드를 입력하세요"
              value={inputCode}
              onChange={(e) => setInputCode(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault(); // ✅ 이 한 줄로 해결!
                  handleLogin();
                }
              }}
              
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
                onClick={handleLogin}
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
