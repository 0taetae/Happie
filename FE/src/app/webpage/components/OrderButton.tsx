"use client";

import { useState } from "react";

export default function OrderButton() {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedRobot, setSelectedRobot] = useState(""); // 선택된 로봇
  const [selectedLocation, setSelectedLocation] = useState(""); // 선택된 장소
  const [selectedTask, setSelectedTask] = useState(""); // 선택된 할 일

  return (
    <div>
      {/* 오더 버튼 */}
      <button
        className="px-6 py-2 bg-green-500 text-white font-bold rounded-lg shadow-lg hover:bg-green-700 transition"
        onClick={() => setIsOpen(true)} // 모달 열기
      >
        ORDER
      </button>

      {/* 모달 */}
      {isOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/50">
          <div
            className="bg-white p-6 rounded-lg shadow-lg w-[400px] relative
            transition-all duration-300 ease-out transform
            scale-100 opacity-100 animate-fadeInModal"
          >
            <h2 className="text-lg font-bold mb-4">로봇 및 명령 선택</h2>

            {/* 🔹 로봇 선택 */}
            <label className="block mb-2 font-semibold">로봇 선택</label>
            <select className="w-full p-2 border rounded mb-4" value={selectedRobot} onChange={(e) => setSelectedRobot(e.target.value)}>
              <option value="">로봇을 선택하세요</option>
              <option value="ROBOT_1">ROBOT_1</option>
              <option value="ROBOT_2">ROBOT_2</option>
              <option value="ROBOT_3">ROBOT_3</option>
            </select>

            {/* 🔹 장소 선택 */}
            <label className="block mb-2 font-semibold">장소 선택</label>
            <select className="w-full p-2 border rounded mb-4" value={selectedLocation} onChange={(e) => setSelectedLocation(e.target.value)}>
              <option value="">장소를 선택하세요</option>
              <option value="병실 1">병실 1</option>
              <option value="약제실">약제실</option>
              <option value="대기실">대기실</option>
            </select>

            {/* 🔹 할일 선택 */}
            <label className="block mb-2 font-semibold">할 일 선택</label>
            <select className="w-full p-2 border rounded mb-4" value={selectedTask} onChange={(e) => setSelectedTask(e.target.value)}>
              <option value="">할 일을 선택하세요</option>
              <option value="링거 전달">링거 전달</option>
              <option value="휠체어 이동">휠체어 이동</option>
              <option value="물품 배송">물품 배송</option>
            </select>

            {/* 버튼 그룹 */}
            <div className="flex justify-end space-x-2">
              <button
                className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500 transition"
                onClick={() => setIsOpen(false)} // 모달 닫기
              >
                취소
              </button>
              <button
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-700 transition"
                onClick={() => {
                  if (!selectedRobot || !selectedLocation || !selectedTask) {
                    alert("모든 항목을 선택해주세요!");
                    return;
                  }
                  console.log(`🚀 ${selectedRobot}에게 "${selectedTask}" 명령을 수행하도록 요청! (${selectedLocation}에서)`);
                  setIsOpen(false);
                }}
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
