"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, useMemo } from "react";
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
  const router = useRouter();
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [statuses, setStatuses] = useState<Record<number, InProgress>>({});
  const [mapImage, setMapImage] = useState<string | null>(null);
  const [robot1Position, setRobot1Position] = useState<{ x: number; y: number } | null>(null);
  const [mapParams, setMapParams] = useState<{ MAP_RESOLUTION: number; MAP_SIZE: number[] } | null>(null);

  const mapWidthPx = useMemo(() => {
    return mapParams ? mapParams.MAP_SIZE[0] / mapParams.MAP_RESOLUTION : 1024;
  }, [mapParams]);

  const mapHeightPx = useMemo(() => {
    return mapParams ? mapParams.MAP_SIZE[1] / mapParams.MAP_RESOLUTION : 1024;
  }, [mapParams]);

  useEffect(() => {
    const fetchStatuses = async () => {
      const newStatuses: Record<number, InProgress> = {};
      await Promise.all([1, 2, 3].map(async (i) => {
        try {
          const res = await fetch(`https://j12e103.p.ssafy.io/api/equipment/order-inprogress/robot${i}`);
          if (!res.ok) throw new Error("API 실패");
          const data = await res.json();
          newStatuses[i] = data;
        } catch (err) {
          console.error(`로봇 ${i} 상태 가져오기 실패`, err);
        }
      }));
      setStatuses(newStatuses);
    };

    fetchStatuses();
  }, []);

  useEffect(() => {
    const handleMapMessage = (topic: string, message: Buffer) => {
      try {
        if (topic === "map/data") {
          const parsed = JSON.parse(message.toString());
          if (parsed.image) setMapImage(`data:image/png;base64,${parsed.image}`);
          if (parsed.params) setMapParams(parsed.params);
        }

        if (topic === "robot/map_position") {
          const [xStr, yStr] = message.toString().split(",");
          const x = parseFloat(xStr);
          const y = parseFloat(yStr);
          if (!isNaN(x) && !isNaN(y)) {
            setRobot1Position({ x, y });
          }
        }
      } catch (err) {
        console.error("❌ MQTT 메시지 파싱 오류:", err);
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

  const staticPositions: Position[] = [
    { id: 2, x: 84, y: 47 },
    { id: 3, x: 92, y: 47 },
  ];

  return (
    <div className="w-full h-full flex flex-col px-4 bg-white">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl text-blue-600">🤖 로봇들의 실시간 위치</h2>
        <OrderButton onOrderSuccess={onOrderSuccess} />
      </div>

      <div className="flex-grow flex items-center justify-center">
        <div className="relative w-full max-w-[550px] aspect-square rounded-lg border border-gray-300 bg-gray-100">
          {mapImage ? (
            <img src={mapImage} alt="Map" className="absolute inset-0 object-contain w-full h-full" />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-3xl">
              🕓 지도를 불러오는 중
              <span className="ml-1"><DotAnimation /></span>
            </div>
          )}

          {/* ✅ 로봇1 마커 */}
          {robot1Position && (
            <RobotMarker
              id={1}
              x={(robot1Position.x / mapWidthPx) * 100}
              y={(robot1Position.y / mapHeightPx) * 100}
              hoveredId={hoveredId}
              setHoveredId={setHoveredId}
              status={statuses[1]}
              onClick={() => router.push("/webpage/bot1")}
            />
          )}

          {/* ✅ 로봇 2~3 마커 (고정 위치) */}
          {staticPositions.map((pos) => (
            <RobotMarker
              key={pos.id}
              id={pos.id}
              x={pos.x}
              y={pos.y}
              hoveredId={hoveredId}
              setHoveredId={setHoveredId}
              status={statuses[pos.id]}
              onClick={() => router.push(`/webpage/bot${pos.id}`)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function RobotMarker({
  id,
  x,
  y,
  hoveredId,
  setHoveredId,
  status,
  onClick,
}: {
  id: number;
  x: number;
  y: number;
  hoveredId: number | null;
  setHoveredId: (id: number | null) => void;
  status?: InProgress;
  onClick: () => void;
}) {
  return (
    <div
      className="absolute z-20 cursor-pointer-custom"
      style={{
        left: `${x}%`,
        top: `${y}%`,
        transform: "translate(-50%, -50%)",
      }}
      onMouseEnter={() => setHoveredId(id)}
      onMouseLeave={() => setHoveredId(null)}
      onClick={onClick}
    >
      <div className="relative flex flex-col items-center">
        <AnimatePresence>
          {hoveredId === id && (
            <motion.div
              className="absolute -top-20 whitespace-nowrap bg-white border border-gray-300 px-3 py-2 rounded-lg text-sm shadow-md z-30"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.2 }}
            >
              <div className="font-semibold text-lg flex items-center gap-1">
                🤖 <span>로봇 {id}</span>
              </div>
              <div className="text-lg text-gray-500">
                {status?.todo?.includes("충전") || status?.todo?.includes("수리")
                  ? status.todo
                  : `${status?.todo ?? "로딩"}하는 중...`}
              </div>
              <div className="absolute left-1/2 -bottom-2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-l-transparent border-r-transparent border-t-white" />
            </motion.div>
          )}
        </AnimatePresence>
        <div className="w-5 h-5 bg-red-500 rounded-full border-2 border-white shadow-md" />
      </div>
    </div>
  );
}
