"use client";
import { usePathname, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import RobotList from "../home/RobotList";
import BotCamera from "./BotCamera";
import BotHistory from "./BotHistory";

interface SidebarProps {
  refreshTrigger: number;
}

export default function Sidebar({ refreshTrigger }: SidebarProps) {
  const router = useRouter();
  const pathname = usePathname();
  const tabs = ["home", "bot1", "bot2", "bot3"];
  const currentTab = pathname.split("/").pop() || "home";

  return (
    <div className="h-full flex flex-col bg-white">
      {/* 🔹 상단 탭 버튼 – 고정 영역 */}
      <div className="flex-shrink-0 h-[44px] m-4 mb-4 relative bg-gray-100 shadow-md rounded-md overflow-hidden">
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
        <div className="flex w-full h-full">
          {tabs.map((tab, index) => (
            <button
              key={index}
              onClick={() => router.push(`/webpage/${tab}`)}
              className={`flex-1 text-center z-10 relative px-2 py-2 text-xl transition-colors duration-200 ${
                currentTab === tab ? "text-white" : "text-gray-700"
              }`}
            >
              {tab.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* 🔸 아래 콘텐츠 – 스크롤 가능 */}
      <div className="flex-grow overflow-y-auto px-4 pb-4">
        {pathname === "/webpage/home" && (
          <RobotList refreshTrigger={refreshTrigger} />
        )}

        {["bot1", "bot2", "bot3"].includes(currentTab) && (
          <div className="flex flex-col gap-2">
            <div className="h-56">
              <BotCamera
                botId={parseInt(currentTab.replace("bot", "") || "1")}
              />
            </div>
            <div>
              <h3 className="text-2xl text-blue-600 py-2">
                📜 로봇 {currentTab.replace("bot", "")} 활동 내역
              </h3>
            </div>
            <div className="overflow-y-auto max-h-[calc(100vh-300px)] pr-2">
              <BotHistory
                botId={parseInt(currentTab.replace("bot", "") || "1")}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
