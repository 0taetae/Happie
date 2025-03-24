"use client";

import { useState } from "react";

export default function DrugHistoryButton() {
  const [isOpen, setIsOpen] = useState(false);

  const dummyData = Array.from({ length: 20 }, (_, i) => ({
    robot: `ROBOT_${(i % 3) + 1}`,
    drug: "놀바덱스정",
    barcode: "8818888881",
    time: "2025.03.12 11:01:38",
  }));

  return (
    <div>
      {/* 약품내역 버튼 */}
      <button className="px-6 py-2 bg-indigo-400 text-white font-bold rounded-lg shadow-lg hover:bg-indigo-500 transition" onClick={() => setIsOpen(true)}>
        약품내역
      </button>

      {/* 모달 */}
      {isOpen && (
        <div
          className="fixed inset-0 flex items-center justify-center bg-black/50 z-50"
          onClick={() => setIsOpen(false)} // ✅ 뒷배경 클릭 시 닫힘
        >
          <div
      className={`
        bg-white p-6 rounded-lg shadow-lg w-[600px] max-h-[80vh] relative flex flex-col
        transition-all duration-300 ease-out transform
        scale-100 opacity-100
        animate-fadeInModal
      `}
      onClick={(e) => e.stopPropagation()}
    >
            {/* 닫기 버튼 */}
            <button className="absolute top-2 right-2 text-xl font-bold text-gray-500 hover:text-black" onClick={() => setIsOpen(false)}>
              ×
            </button>

            {/* 제목 */}
            <h2 className="text-xl font-bold mb-4 text-center">🩺 로봇의 의약품 이동 내역</h2>

            {/* 테이블 고정 헤더 */}
            <table className="w-full text-sm border-t border-b border-black ">
              <thead className="bg-white z-10 border-b border-black">
                <tr>
                  <th className="py-2">이동 로봇</th>
                  <th className="">의약품명</th>
                  <th className="">바코드 번호</th>
                  <th className="">입고 시간</th>
                </tr>
              </thead>
            </table>

            {/* tbody만 스크롤 영역으로 분리 */}
            <div className="overflow-y-auto max-h-[50vh]">
              <table className="w-full text-sm">
                <tbody>
                  {dummyData.map((item, idx) => (
                    <tr key={idx} className="text-center border-t">
                      <td className="py-2 text-blue-600 font-semibold">{item.robot}</td>
                      <td>{item.drug}</td>
                      <td>{item.barcode}</td>
                      <td className="whitespace-pre-line">
                        {item.time.split(" ")[0]}
                        <br />
                        {item.time.split(" ")[1]}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
