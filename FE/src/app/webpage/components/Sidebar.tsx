"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-64 bg-gray-100 h-full p-4 shadow-lg">
      {/* 네비게이션 버튼 */}
      <div className="flex space-x-2">
        {["home", "bot1", "bot2", "bot3"].map((page, index) => (
          <Link key={index} href={`/webpage/${page}`} className={`px-3 py-1 text-sm rounded-lg ${pathname.includes(page) ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-700"}`}>
            {page.toUpperCase()}
          </Link>
        ))}
      </div>

      {/* 로봇 상태 정보 */}
      <div className="mt-6">
        <h2 className="bg-yellow-400 text-white text-center py-2 rounded-lg">의약품 적재 상태</h2>
        
        {[1, 2, 3].map((num) => (
          <div key={num} className="bg-white p-3 my-3 rounded-lg shadow-md">
            <p className="font-bold">🤖 ROBOT_{num}</p>
            <p className="text-sm text-gray-500">{num === 1 ? "환자 물품 배달 중" : num === 2 ? "운행 중" : "충전 중"}</p>
            <div className="mt-2 h-2 bg-gray-200 rounded">
              <div className={`h-full ${num === 1 ? "bg-green-500 w-[80%]" : num === 2 ? "bg-orange-400 w-[53%]" : "bg-red-500 w-[27%]"}`}></div>
            </div>
            <p className="text-sm mt-1">{num === 1 ? "80%" : num === 2 ? "53%" : "27%"}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
