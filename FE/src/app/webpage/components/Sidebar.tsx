"use client";
import { usePathname, useRouter } from "next/navigation";
import RobotList from "../home/RobotList";
import BotCamera from "./BotCamera";
import BotHistory from "./BotHistory";

export default function Sidebar() {
  const router = useRouter();
  const pathname = usePathname(); // 현재 URL 경로 확인

  return (
    <div className="w-72 bg-white h-[90vh] p-4 rounded-xl shadow-lg flex flex-col">
      {/* 네비게이션 바 */}
      <div className="flex w-full mb-4">
        {["home", "bot1", "bot2", "bot3"].map((tab, index) => (
          <button
            key={index}
            onClick={() => router.push(`/webpage/${tab}`)} // 🔹 클릭하면 해당 페이지로 이동
            className={`flex-1 text-center px-2 py-2 border border-gray-300 text-sm font-semibold rounded-md ${
              pathname.includes(tab) ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-300"
            }`}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* 홈 화면: 로봇 리스트 */}
      {pathname === "/webpage/home" && <RobotList />}

      {/* 개별 로봇 페이지 */}
      {["bot1", "bot2", "bot3"].includes(pathname.split("/").pop() || "") && (
        <div className="flex flex-col flex-grow">
          {/* ✅ BotCamera는 고정된 위치로 설정 */}
          <div className="h-36 flex-shrink-0 mb-2">
            <BotCamera botId={parseInt(pathname.split("/").pop()?.replace("bot", "") || "1")} />
          </div>

          {/* ✅ 활동 내역 헤더도 고정 */}
          <div className="flex-shrink-0">
            <h3 className="text-md font-semibold text-blue-600 py-2">
              📜 로봇 {pathname.split("/").pop()?.replace("bot", "")} 활동 내역
            </h3>
          </div>

          {/* ✅ 활동 내역 리스트만 스크롤 가능 */}
          <div className="flex-grow overflow-y-auto">
            <BotHistory botId={parseInt(pathname.split("/").pop()?.replace("bot", "") || "1")} />
          </div>
        </div>
      )}
    </div>
  );
}
