"use client";

import "../globals.css";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Swal from "sweetalert2";
import Sidebar from "./components/Sidebar";
import Map from "./components/Map";
import OrderButton from "./components/OrderButton";
import DrugHistoryButton from "./components/DrugHistoryButton";
import Warning from "./components/Warning";

export default function WebPageLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [isChecking, setIsChecking] = useState(true);
  const [unauthorized, setUnauthorized] = useState(false);

  useEffect(() => {
    const code = localStorage.getItem("access_code");

    if (code !== "103") {
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

  if (isChecking) return null;

  if (unauthorized) {
    return <div className="min-h-screen bg-white" />; // 아무것도 안 보여주기 (모달만 뜸)
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
                <Warning />
                <DrugHistoryButton />
                <OrderButton />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
