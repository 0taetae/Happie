// "use client";
// import { useState } from "react";

// export default function HistoryModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
//   if (!isOpen) return null;

//   return (
//     <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
//       <div className="bg-white p-6 rounded shadow-lg">
//         <h2 className="text-xl font-bold">📜 히스토리</h2>
//         <p>이곳에서 기록을 확인할 수 있습니다.</p>
//         <button className="mt-4 px-4 py-2 bg-red-500 text-white" onClick={onClose}>
//           닫기
//         </button>
//       </div>
//     </div>
//   );
// }
