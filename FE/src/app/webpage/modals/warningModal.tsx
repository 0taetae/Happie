// "use client";

// export default function WarningModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
//   if (!isOpen) return null;

//   return (
//     <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
//       <div className="bg-yellow-100 p-6 rounded shadow-lg">
//         <h2 className="text-xl font-bold">⚠️ 경고</h2>
//         <p>경고 메시지가 표시됩니다.</p>
//         <button className="mt-4 px-4 py-2 bg-red-500 text-white" onClick={onClose}>
//           닫기
//         </button>
//       </div>
//     </div>
//   );
// }
