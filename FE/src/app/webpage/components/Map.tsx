"use client";
import { usePathname } from "next/navigation";

export default function Map() {
  const pathname = usePathname();

  // 현재 페이지에 따라 타이틀 변경
  let title = "로봇들의 실시간 위치"; // 기본값 (홈)
  let content = "[실시간 지도 영역]"; // 기본 텍스트

  if (pathname.includes("bot1")) {
    title = "로봇 1 이동 경로";
    content = "[로봇 1의 이동 경로 표시]";
  } else if (pathname.includes("bot2")) {
    title = "로봇 2 이동 경로";
    content = "[로봇 2의 이동 경로 표시]";
  } else if (pathname.includes("bot3")) {
    title = "로봇 3 이동 경로";
    content = "[로봇 3의 이동 경로 표시]";
  }

  return (
    <div className="w-full max-w-5xl p-6 bg-white rounded-lg shadow-md flex-grow">
      <h2 className="text-xl font-semibold text-blue-600 mb-4">🤖 {title}</h2>
      <div className="h-[430px] flex items-center justify-center rounded-lg border border-gray-300 bg-gray-100">
        <p className="text-gray-500 text-lg font-medium">{content}</p>
      </div>
    </div>
  );
}
