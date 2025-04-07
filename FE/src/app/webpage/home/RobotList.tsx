"use client";

import { useEffect, useState, useCallback } from "react";
import Swal from "sweetalert2";
import "sweetalert2/dist/sweetalert2.min.css";

interface Order {
  id: number;
  robot: string;
  place: string;
  todo: string;
  status: string;
}

interface InProgress {
  robot: string;
  place: string;
  todo: string;
  state: string;
  id: number;
}

const progressBar: Record<number, { percent: number; color: string }> = {
  1: { percent: 80, color: "bg-green-500" },
  2: { percent: 12, color: "bg-red-500" },
  3: { percent: 53, color: "bg-orange-400" },
};

export default function RobotList({ refreshTrigger }: { refreshTrigger: number }) {
  const [openRobot, setOpenRobot] = useState<number | null>(null);
  const [robotTasks, setRobotTasks] = useState<Record<number, Order[]>>({
    1: [],
    2: [],
    3: [],
  });
  const [inProgress, setInProgress] = useState<Record<number, InProgress>>({
    1: { robot: "", place: "", todo: "", state: "로딩 중...", id: 0 },
    2: { robot: "", place: "", todo: "", state: "로딩 중...", id: 0 },
    3: { robot: "", place: "", todo: "", state: "로딩 중...", id: 0 },
  });

  const fetchAllTasks = useCallback(async () => {
    for (let i = 1; i <= 3; i++) {
      const robotKey = `robot${i}`;

      try {
        const ordersRes = await fetch(`https://j12e103.p.ssafy.io/api/equipment/orders/${robotKey}`);
        if (!ordersRes.ok) throw new Error("명령 불러오기 실패");
        const ordersData = await ordersRes.json();
        setRobotTasks((prev) => ({ ...prev, [i]: ordersData }));

        const progressRes = await fetch(`https://j12e103.p.ssafy.io/api/equipment/order-inprogress/${robotKey}`);
        if (!progressRes.ok) throw new Error("상태 불러오기 실패");
        const progressData = await progressRes.json();
        setInProgress((prev) => ({ ...prev, [i]: progressData }));
      } catch (error) {
        console.error(`로봇 ${i} 데이터 불러오기 실패:`, error);
      }
    }
  }, []);

  useEffect(() => {
    fetchAllTasks();
  }, [refreshTrigger, fetchAllTasks]);

  const handleDelete = async (id: number, robotNum: number) => {
    try {
      const res = await fetch(`https://j12e103.p.ssafy.io/api/equipment/order/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("삭제 실패");
      setRobotTasks((prev) => ({
        ...prev,
        [robotNum]: prev[robotNum].filter((task) => task.id !== id),
      }));
    } catch (err) {
      console.error("삭제 실패:", err);
    }
  };

  return (
    <div className="flex flex-col overflow-y-auto h-full gap-2">
      {[1, 2, 3].map((num) => (
        <div key={num} className="bg-gray-100 p-3 mb-2 rounded-lg shadow-md">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">🤖</span>
            <p className="text-2xl text-blue-600">로봇{num}</p>
          </div>
          <p className="text-lg text-gray-600">
  ⭐
            {inProgress[num]?.todo ? (
              /(충전 중|수리 중)$/.test(inProgress[num].todo) ? (
                inProgress[num].todo
              ) : (
                `${inProgress[num].place}에 ${inProgress[num].todo}하는 중...`
              )
            ) : (
              "노는 중.."
            )}
          </p>


          <button className="mt-2 px-3 py-1 text-ml bg-gray-200 rounded-md shadow-sm hover:bg-gray-300 transition" onClick={() => setOpenRobot(openRobot === num ? null : num)}>
            {openRobot === num ? "닫기" : robotTasks[num]?.length > 0 ? `대기 명령 ${robotTasks[num].length}개` : "대기 명령 없음"}
          </button>

          <div
            className={`transition-all duration-300 ease-in-out overflow-hidden mt-2 
              bg-yellow-100 shadow-md rounded-lg text-ml 
              ${openRobot === num ? "max-h-[300px] p-2" : "max-h-0 p-0"}`}
          >
            <div className="pr-1">
              {robotTasks[num]?.length > 0 ? (
                <>
                  <p className="mb-1 text-lg">📌 대기 명령</p>
                  <ul className="space-y-1">
                    {robotTasks[num].map((task, index) => (
                      <li key={task.id} className="px-2 py-1 border-b-2 border-gray-400 flex justify-between items-center">
                        <span>
                          {index + 1}. [{task.place}] {task.todo}하기
                        </span>
                        <button
                          onClick={() => {
                            Swal.fire({
                              title: "정말 삭제하시겠습니까?",
                              // text: "삭제된 명령은 복구할 수 없습니다.",
                              icon: "warning",
                              showCancelButton: true,
                              confirmButtonColor: "#d33",
                              cancelButtonColor: "#3085d6",
                              confirmButtonText: "삭제",
                              cancelButtonText: "취소",
                            }).then((result) => {
                              if (result.isConfirmed) {
                                handleDelete(task.id, num);
                                Swal.fire({
                                  title: "삭제 완료",
                                  text: "해당 명령이 삭제되었습니다.",
                                  icon: "success",
                                  timer: 1000,
                                  showConfirmButton: false,
                                });
                              }
                            });
                          }}
                          className="ml-2 px-2 py-0.5 text-red-600 text-sm rounded hover:font-bold hover:text-red-700 transition-all duration-150"
                        >
                          삭제
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p className="text-gray-700 text-lg">📢 명령을 내려주세요!</p>
              )}
            </div>
          </div>

          <div className="mt-3 flex items-center space-x-2">
            <div className="flex-1 h-2 bg-gray-200 rounded">
              <div className={`h-full rounded ${progressBar[num].color}`} style={{ width: `${progressBar[num].percent}%` }}></div>
            </div>
            <p className="text-sm w-10 text-right">{progressBar[num].percent}%</p>
          </div>
        </div>
      ))}
    </div>
  );
}
