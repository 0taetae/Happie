"use client";

import { usePathname, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import RobotList from "../home/RobotList";
import BotCamera from "./BotCamera";
import BotHistory from "./BotHistory";

export default function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();

  const tabs = ["home", "bot1", "bot2", "bot3"];
  const currentTab = pathname.split("/").pop() || "home";

  return (
    <div className="w-96 bg-white h-[93vh] p-4 flex flex-col">
      {/* 네비게이션 바 */}
      <div className="relative flex w-full mb-6 bg-gray-100 rounded-md overflow-hidden">
        {/* 슬라이딩 배경 */}
        <motion.div
  className={`absolute top-0 bottom-0 rounded-md z-0 ${
    currentTab === "home" ? "bg-green-500" : "bg-blue-500"
  }`}
  layoutId="tab-indicator"
  initial={false}
  transition={{ type: "spring", stiffness: 500, damping: 30 }}
  style={{
    width: `${100 / tabs.length}%`,
    left: `${(tabs.indexOf(currentTab) / tabs.length) * 100}%`,
  }}
/>


        {/* 탭 버튼 */}
        {tabs.map((tab, index) => (
          <button
            key={index}
            onClick={() => router.push(`/webpage/${tab}`)}
            className={`flex-1 text-center z-10 relative px-2 py-2 text-sm font-semibold transition-colors duration-200 ${
              currentTab === tab ? "text-white" : "text-gray-700"
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
          <div className="h-56 flex-shrink-0 mb-2">
            <BotCamera botId={parseInt(pathname.split("/").pop()?.replace("bot", "") || "1")} />
          </div>

          <div className="flex-shrink-0">
            <h3 className="text-md font-semibold text-blue-600 py-2">
              📜 로봇 {pathname.split("/").pop()?.replace("bot", "")} 활동 내역
            </h3>
          </div>

          <div className="flex-grow overflow-y-auto max-h-[265px]">
            <BotHistory botId={parseInt(pathname.split("/").pop()?.replace("bot", "") || "1")} />
          </div>
        </div>
      )}
    </div>
  );
}
