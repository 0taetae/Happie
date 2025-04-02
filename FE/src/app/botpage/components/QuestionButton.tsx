"use client";

import { sendMessage } from "../hooks/useChatbotResponse";

interface Props {
  text: string;
  setQuestion: (q: string) => void;
  setAnswer: (a: string) => void;
  setStage: (s: "idle" | "recording" | "loading" | "answering") => void;
}

export default function QuestionButton({ text, setQuestion, setAnswer, setStage }: Props) {
  const handleClick = () => {
    setQuestion(text);
    setAnswer("");
    setStage("answering");
    sendMessage(text);
  };

  return (
    <button
      onClick={handleClick}
      className="bg-blue-100 hover:bg-blue-200 text-blue-800 font-semibold px-4 py-2 rounded-full shadow transition"
    >
      {text}
    </button>
  );
}
