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
  const [unauthorized, setUnauthorized] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [warningImage, setWarningImage] = useState("");

  // ✅ RobotList 갱신 트리거 상태
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

      setUnauthorized(true);
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
          setWarningImage(data.image_url); // 🔹 이미지 URL 저장
          setShowWarning(true);
        } catch (err) {
          console.error("❌ JSON 파싱 오류:", err);
        }
      }
    });
  }, []);

  if (isChecking) return null;

  if (unauthorized) {
    return <div className="min-h-screen bg-white" />;
  }

  return (
    <div className="flex flex-col h-screen">
      <header className="bg-blue-200 flex justify-between p-4 text-lg font-bold shadow-md">
        <Link href="/" className="text-white">
          🏥 하피 (happie)
        </Link>
        <div className="text-white">한살차이</div>
      </header>

      {/* ⚙️ 콘텐츠 */}
      <div className="flex flex-grow overflow-hidden">
        {/* ✅ Sidebar에 refreshTrigger 전달 */}
        <Sidebar refreshTrigger={refreshTrigger} />

        {/* 콘텐츠 전체 영역 */}
        <div className="flex flex-col flex-grow bg-white relative">
          {/* 📍 지도 - ✅ setRefreshTrigger 함수 전달 */}
          <Map onOrderSuccess={() => setRefreshTrigger((prev) => prev + 1)} />

          {/* 📄 기타 콘텐츠 */}
          <div className="mt-6">{children}</div>
        </div>
      </div>

      {/* 낙상 경고 모달 */}
      {showWarning && <Warning imageUrl={warningImage} onClose={() => setShowWarning(false)} />}
    </div>
  );
}
