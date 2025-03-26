"use client";
import Image from "next/image";
import { useEffect, useState } from "react";

export default function Loading() {
  const [dots, setDots] = useState(".");

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? "." : prev + "."));
    }, 500); // 0.5초마다 바뀜

    return () => clearInterval(interval); // cleanup
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-white">
      <Image
        src="/images/loading.gif"
        alt="로딩 중"
        width={500}
        height={500}
      />
      <p className="mt-4 text-gray-700 font-semibold text-lg">
        하피가 준비중입니다{dots}
      </p>
    </div>
  );
}
