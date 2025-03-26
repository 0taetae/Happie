"use client";
import { useState } from "react";

// 로봇 하나의 데이터 타입 정의
interface Task {
  id: number;
  task: string;
  color: string;
}

interface RobotInfo {
  status: string;
  progress: number;
  progressColor: string;
  tasks: Task[];
}

// 더미 데이터 (각 로봇의 상태 및 대기 임무)
const robotData: Record<number, RobotInfo> = {
  1: {
    status: "링거 폴대를 가지고 병실1로 이동 중...",
    progress: 80,
    progressColor: "bg-green-500",
    tasks: [
      { id: 1, task: "약제실로 이동", color: "bg-orange-300" },
      { id: 2, task: "휠체어를 병실1로 이동", color: "bg-green-300" },
      { id: 3, task: "병실1에서 링거 폴대 전달", color: "bg-red-400" },
    ],
  },
  2: {
    status: "운행 중...",
    progress: 53,
    progressColor: "bg-orange-400",
    tasks: [
      { id: 1, task: "물품 배송 시작", color: "bg-blue-400" },
      { id: 2, task: "병실3으로 이동", color: "bg-yellow-400" },
    ],
  },
  3: {
    status: "충전 중...",
    progress: 27,
    progressColor: "bg-red-500",
    tasks: [],
  },
};

export default function RobotList() {
  const [openRobot, setOpenRobot] = useState<number | null>(null);

  return (
    <div className="flex flex-col overflow-y-auto h-full">
      {[1, 2, 3].map((num) => (
        <div key={num} className="bg-gray-100 p-3 mb-2 rounded-lg shadow-md">
          <div className="flex items-center space-x-3">
            <span className="text-lg">🤖</span>
            <p className="font-bold text-blue-600">ROBOT_{num}</p>
          </div>
          <p className="text-sm text-gray-500">{robotData[num].status}</p>

          {/* "대기 임무 ?개" 버튼 */}
          <button className="mt-2 px-3 py-1 text-sm bg-gray-200 rounded-md shadow-sm hover:bg-gray-300 transition" onClick={() => setOpenRobot(openRobot === num ? null : num)}>
            {openRobot === num ? "닫기" : robotData[num].tasks.length > 0 ? `대기 임무 ${robotData[num].tasks.length}개` : "대기 임무 없음"}
          </button>

          {/* 대기 임무 (토글) */}
          {/* 슬라이드 애니메이션 박스 */}
          <div
            className={`
              transition-all duration-300 ease-in-out overflow-hidden mt-2 
              bg-white shadow-md rounded-lg text-sm 
              ${openRobot === num ? "max-h-[300px] p-2" : "max-h-0 p-0"}
            `}
          >
            {/* 안의 내용은 항상 렌더링되지만 숨기기 위해 max-height 활용 */}
            <div className="overflow-y-auto max-h-[250px] pr-1">
              {robotData[num].tasks.length > 0 ? (
                <>
                  <p className="mb-1 font-bold">📌 대기 임무</p>
                  <ul className="space-y-1">
                    {robotData[num].tasks.map((task) => (
                      <li key={task.id} className={`${task.color} px-2 py-1 rounded-md text-white`}>
                        {task.id}. {task.task}
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p className="text-gray-500 font-semibold">📢 임무를 내려주세요!</p>
              )}
            </div>
          </div>

          {/* 프로그레스 바 */}
          <div className="mt-3 h-2 bg-gray-200 rounded">
            <div className={`h-full rounded ${robotData[num].progressColor}`} style={{ width: `${robotData[num].progress}%` }}></div>
          </div>
          <p className="text-sm mt-1 text-right font-semibold">{robotData[num].progress}%</p>
        </div>
      ))}
    </div>
  );
}
