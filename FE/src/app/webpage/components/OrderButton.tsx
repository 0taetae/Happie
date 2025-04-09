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
    "501호실": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "502호실": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "503호실": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "간호사실": ["출동하기"],
    "병동": ["운행하기", "청소하기"],
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
        title: "모든 항목을 선택해주세요!😢",
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
        title: "명령을 시켰어요.☺️",
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
        className="px-6 py-2 bg-green-500 text-white text-xl rounded-lg shadow-lg hover:bg-green-700 transition"
        onClick={() => {
          setIsOpen(true);
          resetSelections();
        }}
      >
        ORDER
      </button>

      {isOpen && (
        <div
          className="cursor-default-custom fixed inset-0 flex items-center justify-center bg-black/50 z-50"
          onClick={() => {
            setIsOpen(false);
            resetSelections();
          }}
        >
          <div
            className="cursor-default-custom bg-white p-14 rounded-lg shadow-lg w-[600px] relative transition-all duration-300 ease-out transform scale-100 opacity-100 animate-fadeInModal"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-center text-cyan-600 text-4xl mb-8">로봇 및 명령 선택</h2>

            {/* 로봇 선택 */}
            <label className="cursor-default-custom text-3xl block mb-2">로봇 선택</label>
            <select
              className="cursor-default-custom w-full p-2 text-xl border rounded mb-8"
              value={selectedRobot}
              onChange={(e) => setSelectedRobot(e.target.value)}
            >
              <option value="">로봇을 선택하세요</option>
              <option value="robot1">로봇1</option>
              <option value="robot2">로봇2</option>
              <option value="robot3">로봇3</option>
            </select>

            {/* 장소 선택 */}
            <label className="cursor-default-custom text-3xl block mb-2">장소 선택</label>
            <select
              className="cursor-default-custom w-full p-2 text-xl border rounded mb-8"
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
            <label className="cursor-default-custom text-3xl block mb-2">할 일 선택</label>
            <select
              className="cursor-default-custom w-full p-2 text-xl border rounded mb-8"
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
                className="px-4 py-2 text-xl bg-cyan-600 text-white rounded hover:bg-cyan-700 transition"
                onClick={handleConfirm}
              >
                확인
              </button>
              <button
                className="px-4 py-2 text-xl bg-gray-400 text-white rounded hover:bg-gray-500 transition"
                onClick={() => {
                  setIsOpen(false);
                  resetSelections();
                }}
              >
                취소
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
