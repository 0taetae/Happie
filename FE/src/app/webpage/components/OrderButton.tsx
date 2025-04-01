"use client";

import { useState } from "react";
import Swal from "sweetalert2";
import "sweetalert2/dist/sweetalert2.min.css";

export default function OrderButton({ onOrderSuccess }: { onOrderSuccess?: () => void }) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedRobot, setSelectedRobot] = useState("");
  const [selectedLocation, setSelectedLocation] = useState("");
  const [selectedTask, setSelectedTask] = useState("");

  const taskOptionsByLocation: Record<string, string[]> = {
    "병실 1": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "병실 2": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "병실 3": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "데스크": ["출동하기"],
    "전체": ["운행하기", "청소하기"],
  };

  const resetSelections = () => {
    setSelectedRobot("");
    setSelectedLocation("");
    setSelectedTask("");
  };

  const handleConfirm = async () => {
    if (!selectedRobot || !selectedLocation || !selectedTask) {
      Swal.fire({
        icon: "warning",
        title: "항목 선택 필요",
        text: "모든 항목을 선택해주세요!",
        confirmButtonColor: "#3085d6",
      });
      return;
    }
  
    const trimmedTask = selectedTask.endsWith("하기")
      ? selectedTask.slice(0, -2) // '하기' 제거
      : selectedTask;
  
    try {
      const response = await fetch(
        "https://j12e103.p.ssafy.io/api/equipment/create-order",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            robot: selectedRobot,
            place: selectedLocation,
            todo: trimmedTask, // ✅ 가공된 값 사용
          }),
        }
      );
  
      if (!response.ok) throw new Error("서버 응답 실패");
      await response.json();
  
      if (onOrderSuccess) onOrderSuccess();
  
      Swal.fire({
        icon: "success",
        title: "명령 입력 완료",
        confirmButtonColor: "#3085d6",
      });
  
      setIsOpen(false);
      resetSelections();
    } catch (error) {
      console.error("명령 전송 실패:", error);
      Swal.fire({
        icon: "error",
        title: "전송 실패",
        text: "서버와의 연결에 실패했습니다. 다시 시도해주세요.",
        confirmButtonColor: "#d33",
      });
    }
  };
  

  return (
    <div>
      <button
        className="px-6 py-2 bg-green-500 text-white font-bold rounded-lg shadow-lg hover:bg-green-700 transition"
        onClick={() => {
          setIsOpen(true);
          resetSelections();
        }}
      >
        ORDER
      </button>

      {isOpen && (
        <div
          className="fixed inset-0 flex items-center justify-center bg-black/50 z-50"
          onClick={() => {
            setIsOpen(false);
            resetSelections();
          }}
        >
          <div
            className="bg-white p-6 rounded-lg shadow-lg w-[400px] relative transition-all duration-300 ease-out transform scale-100 opacity-100 animate-fadeInModal"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-bold mb-4">로봇 및 명령 선택</h2>

            {/* 로봇 선택 */}
            <label className="block mb-2 font-semibold">로봇 선택</label>
            <select
              className="w-full p-2 border rounded mb-4"
              value={selectedRobot}
              onChange={(e) => setSelectedRobot(e.target.value)}
            >
              <option value="">로봇을 선택하세요</option>
              <option value="robot1">로봇1</option>
              <option value="robot2">로봇2</option>
              <option value="robot3">로봇3</option>
            </select>

            {/* 장소 선택 */}
            <label className="block mb-2 font-semibold">장소 선택</label>
            <select
              className="w-full p-2 border rounded mb-4"
              value={selectedLocation}
              onChange={(e) => {
                const selected = e.target.value;
                setSelectedLocation(selected);
                const tasks = taskOptionsByLocation[selected];
                if (tasks?.length === 1) setSelectedTask(tasks[0]);
                else setSelectedTask("");
              }}
            >
              <option value="">장소를 선택하세요</option>
              {Object.keys(taskOptionsByLocation).map((location) => (
                <option key={location} value={location}>
                  {location}
                </option>
              ))}
            </select>

            {/* 할 일 선택 */}
            <label className="block mb-2 font-semibold">할 일 선택</label>
            <select
              className="w-full p-2 border rounded mb-4"
              value={selectedTask}
              onChange={(e) => setSelectedTask(e.target.value)}
              disabled={!selectedLocation}
            >
              <option value="">할 일을 선택하세요</option>
              {selectedLocation &&
                taskOptionsByLocation[selectedLocation]?.map((task) => (
                  <option key={task} value={task}>
                    {task}
                  </option>
                ))}
            </select>

            {/* 버튼 그룹 */}
            <div className="flex justify-end space-x-2">
              <button
                className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500 transition"
                onClick={() => {
                  setIsOpen(false);
                  resetSelections();
                }}
              >
                취소
              </button>
              <button
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700 transition"
                onClick={handleConfirm}
              >
                확인
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
