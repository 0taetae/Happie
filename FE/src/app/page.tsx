"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Swal from "sweetalert2";

export default function Home() {
  const [isLoading, setIsLoading] = useState(true); // ✅ 로딩 상태
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [inputCode, setInputCode] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  useEffect(() => {
    const savedCode = localStorage.getItem("access_code");

    if (savedCode === "gkstkfckdl0411!") {
      router.push("/webpage/home");
    } else {
      setIsLoading(false); // ✅ 로그인 필요: 본 화면 렌더링 허용
    }
  }, [router]);

  useEffect(() => {
    if (isLoginOpen) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  }, [isLoginOpen]);

  const handleLogin = () => {
    if (inputCode === "gkstkfckdl0411!") {
      Swal.fire({
        icon: "success",
        title: "로그인 성공!",
        text: "Happie에 오신 걸 환영합니다 😊",
        showConfirmButton: false,
        timer: 1500,
      });

      setTimeout(() => {
        localStorage.setItem("access_code", "gkstkfckdl0411!");
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

  // ✅ 로그인 체크 중엔 아무 것도 렌더링하지 않음
  if (isLoading) return null;

  return (
    <div className="flex h-screen bg-white">
      {/* 왼쪽 화면 (WEB) */}
      <div
        className="w-1/2 flex flex-col items-center justify-center text-pink-400 text-6xl font-bold cursor-pointer-custom transition-all duration-300 hover:shadow-[0_0_60px_0_rgba(251,113,133,0.5)]"
        onClick={() => setIsLoginOpen(true)}
      >
        <img
          src="/images/webpage.gif"
          alt="WEB GIF"
          style={{ width: "400px", height: "400px" }}
          className="mb-4 object-contain"
        />
        WEB
      </div>

      {/* 오른쪽 화면 (BOT) */}
      <div
        className="w-1/2 flex flex-col items-center justify-center text-green-600 text-6xl font-bold cursor-pointer-custom transition-all duration-300 hover:shadow-[0_0_60px_0_rgba(34,197,94,0.5)]"
        onClick={() => router.push("/botpage")}
      >
        <img
          src="/images/botpage.gif"
          alt="BOT GIF"
          style={{ width: "400px", height: "400px" }}
          className="mb-4 object-contain"
        />
        ROBOT
      </div>

      {/* 로그인 모달 */}
      {isLoginOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/50 z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-80 text-center">
            <h2 className="text-lg font-bold mb-4">로그인</h2>

            <input
              ref={inputRef}
              type="password"
              placeholder="코드를 입력하세요"
              value={inputCode}
              onChange={(e) => setInputCode(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
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
