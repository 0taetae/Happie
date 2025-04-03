"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { mqttClient } from "@/lib/mqttClient"; // ✅ 기존에 쓰던 mqtt client import
import OrderButton from "./OrderButton";

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

  // ✅ MQTT: map/data 토픽 수신
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
        const payload = message.toString(); // 예: "221,231"
        const [x, y] = payload.split(",").map(Number);
        if (!isNaN(x) && !isNaN(y)) {
          console.log("📍 로봇1 위치 수신:", x, y);
          setRobot1Position({ x, y });
        }
      }
    };

    mqttClient.on("message", handleMapMessage);
    mqttClient.subscribe("map/data");
    mqttClient.subscribe("robot/map_position"); // ✅ 로봇 위치 구독

    return () => {
      mqttClient.off("message", handleMapMessage);
      mqttClient.unsubscribe("map/data");
      mqttClient.unsubscribe("robot/map_position");
    };
  }, []);

  const positions: Position[] = [
    { id: 2, x: 80, y: 45 },
    { id: 3, x: 88, y: 45 },
  ];

  const title = pathname.includes("bot1") ? "로봇 1 이동 경로" : pathname.includes("bot2") ? "로봇 2 이동 경로" : pathname.includes("bot3") ? "로봇 3 이동 경로" : "로봇들의 실시간 위치";

  return (
    <div className="w-full h-full flex flex-col px-4 py-4 md:px-6 md:py-6 bg-white">
      <div className="flex items-center justify-between mb-2 md:mb-4">
        <h2 className="text-xl font-semibold text-blue-600">🤖 {title}</h2>
        <OrderButton onOrderSuccess={onOrderSuccess} />
      </div>

      {/* ✅ Map 영역: 남은 공간만 사용 */}
      <div className="flex-grow flex items-center justify-center">
        <div className="relative w-full max-w-[550px] aspect-square rounded-lg border border-gray-300 bg-gray-100 overflow-hidden">
          {/* ✅ 실시간 수신 이미지 */}
          {mapImage ? 
            <img src={mapImage} 
              alt="Map" 
              className="absolute inset-0 object-contain w-full h-full" 
            /> : 
            <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">🕓 지도를 불러오는 중...</div>
            }
          {robot1Position && (
            <div
              className="absolute z-20"
              style={{
                left: `${robot1Position.x}px`,
                top: `${robot1Position.y}px`,
                transform: "translate(-50%, -50%)",
              }}
            >
              <div className="w-5 h-5 bg-green-500 rounded-full border-2 border-white shadow-md" />
            </div>
          )}
          {/* ✅ 로봇 포지션 마커 */}
          {positions.map((pos) => (
            <div
              key={pos.id}
              className="absolute cursor-pointer-custom z-10"
              style={{
                left: `${pos.x}%`,
                top: `${pos.y}%`,
                transform: "translate(-50%, -50%)",
              }}
              onMouseEnter={() => setHoveredId(pos.id)}
              onMouseLeave={() => setHoveredId(null)}
              onClick={() => router.push(`/webpage/bot${pos.id}`)}
            >
              <div className="cursor-pointer-custom w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-md" />

              <AnimatePresence>
                {hoveredId === pos.id && (
                  <motion.div className="cursor-pointer-custom absolute -top-14 left-1/2 -translate-x-1/2 whitespace-nowrap bg-white border border-gray-300 px-3 py-2 rounded-lg text-sm shadow-md z-20" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} transition={{ duration: 0.2 }}>
                    <div className="font-semibold">🤖 로봇 {pos.id}</div>
                    <div className="text-xs text-gray-500">{statuses[pos.id]?.todo?.includes("충전") || statuses[pos.id]?.todo?.includes("수리") ? statuses[pos.id]?.todo : `${statuses[pos.id]?.todo ?? "불러오는 중..."} 하는 중...`}</div>

                    <div className="cursor-pointer-custom absolute left-1/2 -bottom-2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-l-transparent border-r-transparent border-t-white" />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
