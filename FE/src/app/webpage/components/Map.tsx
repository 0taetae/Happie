"use client";

import { usePathname, useRouter } from "next/navigation";
import Image from "next/image";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
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

  const positions: Position[] = [
    { id: 1, x: 45, y: 20 },
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
          <Image src="/images/map.png" alt="Map" fill className="object-contain" />

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
                  <motion.div className="cursor-pointer-custom absolute -top-14 left-1/2 -translate-x-1/2 whitespace-nowrap bg-white border border-gray-300 px-3 py-2 rounded-lg text-sm shadow-md z-20">
                    <div className="font-semibold">🤖 로봇 {pos.id}</div>
                    <div className="text-xs text-gray-500">{statuses[pos.id]?.todo?.includes("충전") || statuses[pos.id]?.todo?.includes("수리") ? statuses[pos.id]?.todo : `${statuses[pos.id]?.todo ?? "불러오는 중..."} 하는 중...`}</div>

                    {/* 화살표 */}
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
