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
    <div className="w-96 bg-white h-full p-4 flex flex-col min-h-0 overflow-hidden">
      <div className="relative flex w-full mb-6 bg-gray-100 rounded-md overflow-hidden">
        <motion.div
          className={`absolute top-0 bottom-0 rounded-md z-0 ${currentTab === "home" ? "bg-green-500" : "bg-blue-500"}`}
          layoutId="tab-indicator"
          initial={false}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
          style={{
            width: `${100 / tabs.length}%`,
            left: `${(tabs.indexOf(currentTab) / tabs.length) * 100}%`,
          }}
        />
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

      {/* 로봇 리스트 - 홈일 경우에만 보여줌 */}
      {pathname === "/webpage/home" && (
        <RobotList refreshTrigger={refreshTrigger} />
      )}

      {/* 개별 로봇 페이지 */}
      {["bot1", "bot2", "bot3"].includes(currentTab) && (
        <div className="flex flex-col flex-grow min-h-0">
          <div className="h-56 flex-shrink-0 mb-2">
            <BotCamera botId={parseInt(currentTab.replace("bot", "") || "1")} />
          </div>
          <div className="flex-shrink-0">
            <h3 className="text-md font-semibold text-blue-600 py-2">
              📜 로봇 {currentTab.replace("bot", "")} 활동 내역
            </h3>
          </div>
          <div className="flex-grow overflow-y-auto min-h-0">
            <BotHistory botId={parseInt(currentTab.replace("bot", "") || "1")} />
          </div>
        </div>
      )}
    </div>
  );
}
