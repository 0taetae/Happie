"use client";

import { useState } from "react";
import Swal from "sweetalert2";
import "sweetalert2/dist/sweetalert2.min.css";

export default function OrderButton() {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedRobot, setSelectedRobot] = useState("");
  const [selectedLocation, setSelectedLocation] = useState("");
  const [selectedTask, setSelectedTask] = useState("");

  // 장소별 할 일 옵션 정의
  const taskOptionsByLocation: Record<string, string[]> = {
    "병실 1": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "병실 2": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "병실 3": ["링거 전달하기", "휠체어 전달하기", "방문하기"],
    "데스크": ["호출하기"],
    "전체": ["청소하기"],
  };

  const resetSelections = () => {
    setSelectedRobot("");
    setSelectedLocation("");
    setSelectedTask("");
  };

  const handleConfirm = () => {
    if (!selectedRobot || !selectedLocation || !selectedTask) {
      Swal.fire({
        icon: "warning",
        title: "항목 선택 필요",
        text: "모든 항목을 선택해주세요!",
        confirmButtonColor: "#3085d6",
      });
      return;
    }

    Swal.fire({
      icon: "success",
      title: "명령 전송 완료",
      text: `${selectedRobot}에게 "${selectedTask}" 명령 요청 완료!`,
      confirmButtonColor: "#3085d6",
    });

    setIsOpen(false);
    resetSelections();
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

            {/* 🔹 로봇 선택 */}
            <label className="block mb-2 font-semibold">로봇 선택</label>
            <select
              className="w-full p-2 border rounded mb-4"
              value={selectedRobot}
              onChange={(e) => setSelectedRobot(e.target.value)}
            >
              <option value="">로봇을 선택하세요</option>
              <option value="ROBOT_1">ROBOT_1</option>
              <option value="ROBOT_2">ROBOT_2</option>
              <option value="ROBOT_3">ROBOT_3</option>
            </select>

            {/* 🔹 장소 선택 */}
            <label className="block mb-2 font-semibold">장소 선택</label>
            <select
              className="w-full p-2 border rounded mb-4"
              value={selectedLocation}
              onChange={(e) => {
                const selected = e.target.value;
                setSelectedLocation(selected);

                const tasks = taskOptionsByLocation[selected];
                // 자동 선택 조건: 해당 장소의 할 일이 하나뿐이면 바로 세팅
                if (tasks && tasks.length === 1) {
                  setSelectedTask(tasks[0]);
                } else {
                  setSelectedTask(""); // 그 외엔 초기화
                }
              }}
            >
              <option value="">장소를 선택하세요</option>
              {Object.keys(taskOptionsByLocation).map((location) => (
                <option key={location} value={location}>
                  {location}
                </option>
              ))}
            </select>

            {/* 🔹 할 일 선택 */}
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
