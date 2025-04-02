"use client";

type Props = {
  onClick: () => void;
};

export default function VoiceButton({ onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className="bg-blue-500 text-white px-6 py-3 rounded-full shadow-lg hover:bg-blue-600 transition text-lg font-semibold"
    >
      🎙️ 음성으로 질문하기
    </button>
  );
}
