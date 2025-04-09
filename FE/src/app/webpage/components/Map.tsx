"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { mqttClient } from "@/lib/mqttClient";
import OrderButton from "./OrderButton";
import DotAnimation from "./DotAnimation";

interface Position {
  id: number;
  x: number;
  y: number;
}

interface InProgress {
  robot: string;
  place: string;
  todo: string;
  state: string;
  id: number;
}

export default function Map({ onOrderSuccess }: { onOrderSuccess: () => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [statuses, setStatuses] = useState<Record<number, InProgress>>({});
  const [mapImage, setMapImage] = useState<string | null>(null);
  const [robot1Position, setRobot1Position] = useState<{ x: number; y: number } | null>(null);

  const fetchStatuses = async () => {
    const newStatuses: Record<number, InProgress> = {};
    for (let i = 1; i <= 3; i++) {
      try {
        const res = await fetch(`https://j12e103.p.ssafy.io/api/equipment/order-inprogress/robot${i}`);
        if (!res.ok) throw new Error("API 실패");
        const data = await res.json();
        newStatuses[i] = data;
      } catch (err) {
        console.error(`로봇 ${i} 상태 가져오기 실패`, err);
      }
    }
    setStatuses(newStatuses);
  };

  useEffect(() => {
    fetchStatuses();
  }, []);

  useEffect(() => {
    const handleMapMessage = (topic: string, message: Buffer) => {
      if (topic === "map/data") {
        try {
          const parsed = JSON.parse(message.toString());
          const base64 = parsed.image;
          setMapImage(`data:image/png;base64,${base64}`);
        } catch (err) {
          console.error("❌ 맵 데이터 파싱 오류:", err);
        }
      }

      if (topic === "robot/map_position") {
        const [x, y] = message.toString().split(",").map(Number);
        if (!isNaN(x) && !isNaN(y)) {
          setRobot1Position({ x, y });
        }
      }
    };

    mqttClient.on("message", handleMapMessage);
    mqttClient.subscribe("map/data");
    mqttClient.subscribe("robot/map_position");

    return () => {
      mqttClient.off("message", handleMapMessage);
      mqttClient.unsubscribe("map/data");
      mqttClient.unsubscribe("robot/map_position");
    };
  }, []);

  const positions: Position[] = [
    { id: 2, x: 71, y: 67 },
    { id: 3, x: 79, y: 67 },
  ];

  const title = pathname.includes("bot1")
    ? "로봇 1 이동 경로"
    : pathname.includes("bot2")
    ? "로봇 2 이동 경로"
    : pathname.includes("bot3")
    ? "로봇 3 이동 경로"
    : "로봇들의 실시간 위치";

  return (
    <div className="w-full h-full flex flex-col px-4 bg-white">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl text-blue-600">🤖 {title}</h2>
        <OrderButton onOrderSuccess={onOrderSuccess} />
      </div>

      <div className="flex-grow flex items-center justify-center">
        <div className="relative w-full max-w-[550px] aspect-square rounded-lg border border-gray-300 bg-gray-100 overflow-hidden">
          {mapImage ? (
            <img src={mapImage} alt="Map" className="absolute inset-0 object-contain w-full h-full" />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-3xl">
              🕓 지도를 불러오는 중
              <span className="ml-1">
                              <DotAnimation />
                            </span>
            </div>
          )}

          {/* ✅ 로봇1 마커 + 툴팁 */}
          {robot1Position && (
            <div
              className="absolute z-20 cursor-pointer-custom"
              style={{
                left: `${(robot1Position.x / 1024) * 100}%`,
                top: `${(robot1Position.y / 1024) * 100}%`,
                transform: "translate(-50%, -50%)",
              }}
              onMouseEnter={() => setHoveredId(1)}
              onMouseLeave={() => setHoveredId(null)}
              onClick={() => router.push("/webpage/bot1")}
            >
              <div className="relative flex flex-col items-center">
                {/* 툴팁 */}
                <AnimatePresence>
                  {hoveredId === 1 && (
                    <motion.div
                      className="absolute -top-20 whitespace-nowrap bg-white border border-gray-300 px-3 py-2 rounded-lg text-sm shadow-md z-30"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}
                      transition={{ duration: 0.2 }}
                    >
                      <div className="font-semibold text-lg flex items-center gap-1">
                        🤖 <span>로봇 1</span>
                      </div>
                      <div className="text-lg text-gray-500">
                        {statuses[1]?.todo?.includes("충전") || statuses[1]?.todo?.includes("수리")
                          ? statuses[1]?.todo
                          : `${statuses[1]?.todo ?? "로딩"}하는 중...`}
                      </div>
                      <div className="absolute left-1/2 -bottom-2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-l-transparent border-r-transparent border-t-white" />
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* 마커 */}
                <div className="w-5 h-5 bg-red-500 rounded-full border-2 border-white shadow-md" />
              </div>
            </div>
          )}

          {/* ✅ 로봇 2, 3 마커 */}
          {positions.map((pos) => (
            <div
              key={pos.id}
              className="absolute cursor-pointer-custom z-20"
              style={{
                left: `${pos.x}%`,
                top: `${pos.y}%`,
                transform: "translate(-50%, -50%)",
              }}
              onMouseEnter={() => setHoveredId(pos.id)}
              onMouseLeave={() => setHoveredId(null)}
              onClick={() => router.push(`/webpage/bot${pos.id}`)}
            >
              <div className="relative flex flex-col items-center">
                <AnimatePresence>
                  {hoveredId === pos.id && (
                    <motion.div
                      className="absolute -top-20 whitespace-nowrap bg-white border border-gray-300 px-3 py-2 rounded-lg text-sm shadow-md z-30"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}
                      transition={{ duration: 0.2 }}
                    >
                      <div className="font-semibold text-lg flex items-center gap-1">
                        🤖 <span>로봇 {pos.id}</span>
                      </div>
                      <div className="text-lg text-gray-500">
                        {statuses[pos.id]?.todo?.includes("충전") || statuses[pos.id]?.todo?.includes("수리")
                          ? statuses[pos.id]?.todo
                          : `${statuses[pos.id]?.todo ?? "로딩"}하는 중...`}
                      </div>
                      <div className="absolute left-1/2 -bottom-2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-l-transparent border-r-transparent border-t-white" />
                    </motion.div>
                  )}
                </AnimatePresence>
                <div className="w-5 h-5 bg-red-500 rounded-full border-2 border-white shadow-md" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
