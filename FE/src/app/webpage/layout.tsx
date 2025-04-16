"use client";

import "../globals.css";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Swal from "sweetalert2";
import Sidebar from "./components/Sidebar";
import Map from "./components/Map";
import Warning from "./components/Warning";
import { mqttClient } from "@/lib/mqttClient";
import Link from "next/link";

export default function WebPageLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [isChecking, setIsChecking] = useState(true);
  // const [unauthorized, setUnauthorized] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [warningImage, setWarningImage] = useState("");

  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    const code = localStorage.getItem("access_code");

    if (code !== "gkstkfckdl0411!") {
      Swal.fire({
        icon: "warning",
        title: "접근 권한 없음",
        text: "코드를 먼저 입력해주세요.",
        showConfirmButton: false,
        timer: 1000,
      });

      // setUnauthorized(true);
      setTimeout(() => {
        router.push("/");
      }, 1000);
    }

    setIsChecking(false);
  }, [router]);

  useEffect(() => {
    mqttClient.on("message", (topic, message) => {
      if (topic === "fall_detection") {
        console.log("📩 낙상 감지 수신:", message.toString());
        try {
          const data = JSON.parse(message.toString());
          setWarningImage(data.image_url);
          setShowWarning(true);
        } catch (err) {
          console.error("❌ JSON 파싱 오류:", err);
        }
      }
    });
  }, []);

  if (isChecking) return null;

  // if (unauthorized) {
  //   return <div className="min-h-screen bg-white" />;
  // }

  return (
    <div className="flex flex-col h-screen font-bmjua bg-white">
      {/* ✅ 일반 흐름의 header (고정 X) */}
      <header className="h-20 bg-blue-200 shadow-md flex justify-between items-center">
        {/* 👈 왼쪽: 로고 + 텍스트 */}
        <Link href="/" className="flex items-center px-12">
          <img
            src="/images/logo.png"
            alt="하피 로고"
            className="w-20 h-20 object-contain"
          />
          <span
  className="ml-2 text-white text-5xl"
  style={{ textShadow: "1px 1px 2px black" }}
>
  HAPPIE
</span>
        </Link>

        {/* 👉 오른쪽 텍스트 */}
        <Link href="/botpage" className="flex items-center">
          <div className="text-white text-4xl px-16" style={{ textShadow: "0.5px 0.5px 1px green" }}>한살차이</div>
        </Link>
      </header>

      {/* ✅ 본문 콘텐츠 영역: 사이드바 + 메인 */}
      <div className="flex flex-row flex-grow h-full overflow-hidden">
        {/* 🔹 Sidebar */}
        <div className="w-96 h-full pl-6 bg-white border-r-2 border-gray-200/60 overflow-y-auto">
          <Sidebar refreshTrigger={refreshTrigger} />
        </div>

        {/* 🔸 Main 콘텐츠 */}
        <div className="flex flex-col flex-grow h-full p-4 bg-white relative overflow-y-auto">
          <Map onOrderSuccess={() => setRefreshTrigger((prev) => prev + 1)} />
          <div className="mt-6">{children}</div>
        </div>
      </div>

      {/* ⚠️ 낙상 감지 경고 */}
      {showWarning && (
        <Warning imageUrl={warningImage} onClose={() => setShowWarning(false)} />
      )}
    </div>
  );
}
