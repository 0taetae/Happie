"use client";

type Props = {
  text: string;
  onClick: (text: string) => void;
};

export default function QuestionButton({ text, onClick }: Props) {
  return (
    <button
      onClick={() => onClick(text)}
      className="bg-white border border-gray-300 rounded-xl px-6 py-4 shadow hover:bg-blue-50 transition text-gray-800 font-medium"
    >
      {text}
    </button>
  );
}
