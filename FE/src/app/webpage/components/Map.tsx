"use client";
import { usePathname } from "next/navigation";
import Image from "next/image";

export default function Map() {
  const pathname = usePathname();

  let title = "로봇들의 실시간 위치";

  if (pathname.includes("bot1")) {
    title = "로봇 1 이동 경로";
  } else if (pathname.includes("bot2")) {
    title = "로봇 2 이동 경로";
  } else if (pathname.includes("bot3")) {
    title = "로봇 3 이동 경로";
  }

  // 더미 좌표 데이터 (x, y는 % 기준)
  const dummyPositions = [
    { id: 1, x: 15, y: 20 },
    { id: 2, x: 50, y: 80 },
    { id: 3, x: 80, y: 30 },
  ];

  return (
    <div className="w-full max-w-5xl p-4 md:p-6  bg-white flex-grow">
      <h2 className="text-xl font-semibold text-blue-600 mb-4">🤖 {title}</h2>

      <div className="relative h-[500px] w-[500px] mx-auto rounded-lg border border-gray-300 bg-gray-100 overflow-hidden">
        <Image src="/images/map.png" alt="Map" width={430} height={430} className="w-full h-full object-contain" />

        {dummyPositions.map((pos) => (
          <div
            key={pos.id}
            className="absolute w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-md"
            style={{
              left: `${pos.x}%`,
              top: `${pos.y}%`,
              transform: "translate(-50%, -50%)",
            }}
          />
        ))}
      </div>
    </div>
  );
}
