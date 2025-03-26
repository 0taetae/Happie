"use client";

interface HelpModalProps {
  isOpen: boolean;
  fadeOut: boolean;
  onClose: () => void;
}

export default function HelpModal({ isOpen, fadeOut, onClose }: HelpModalProps) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50"
      onClick={onClose} // 배경 클릭 시 닫힘
    >
      <div
        className={`bg-white rounded-2xl shadow-2xl p-6 w-[90%] max-w-md relative transition-all animate-fadeInModal ${
          fadeOut ? "fade-out" : "animate-fadeInModal"
        }`}
        onClick={(e) => e.stopPropagation()} // 내부 클릭은 막기
      >
        {/* 우측 상단 닫기 버튼 */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 text-xl font-bold"
          aria-label="닫기"
        >
          &times;
        </button>

        <h2 className="text-xl font-semibold mb-4 text-blue-600">📋 하피가 도와줄 수 있는 질문 리스트!</h2>
        <ul className="text-gray-600 list-disc list-inside space-y-1 mb-6">
          <li>삼성병원에 대해 알려줘.</li>
          <li>502호 병실은 어디에 있어?</li>
          <li>링거폴대 보관실을 찾고 있어.</li>
          <li>간호사실은 어디야?</li>
          <li>내과를 가고 싶어.</li>
          <li>외래약국은 어디니?</li>
        </ul>
      </div>
    </div>
  );
}
