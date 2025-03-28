"use client";

import "../globals.css";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Swal from "sweetalert2";
import Sidebar from "./components/Sidebar";
import Map from "./components/Map";
import OrderButton from "./components/OrderButton";
import Warning from "./components/Warning";
import { mqttClient } from "@/lib/mqttClient";

export default function WebPageLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [isChecking, setIsChecking] = useState(true);
  const [unauthorized, setUnauthorized] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [warningImage, setWarningImage] = useState("");

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
        <div className="text-white">🏥 하피 (happie)</div>
        <div className="text-white">한살차이</div>
      </header>

      <div className="flex flex-grow">
        <Sidebar />
        <main className="flex flex-col flex-grow p-4">
          <div>{children}</div>
          <div className="mt-4 flex flex-col items-center space-y-4">
            <Map />
            <div className="self-end">
              <div className="flex flex-row gap-3">
                <OrderButton />
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* 낙상 경고 모달 */}
      {showWarning && <Warning imageUrl={warningImage} onClose={() => setShowWarning(false)} />}
    </div>
  );
}
