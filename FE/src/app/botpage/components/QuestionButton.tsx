"use client";

import { sendMessage } from "../hooks/useChatbotResponse";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { ColorType } from "../page"; // color type import

interface Props {
  text: string;
  color?: ColorType;
  selected: boolean;
  onSelect: () => void;
  setQuestion: (q: string) => void;
  setAnswer: (a: string) => void;
  setStage: (s: "idle" | "recording" | "loading" | "answering") => void;
}

const colorClassMap: Record<ColorType, string> = {
  amber: "bg-amber-100 border-amber-300 hover:bg-amber-200",
  sky: "bg-sky-100 border-sky-300 hover:bg-sky-200",
  rose: "bg-rose-100 border-rose-300 hover:bg-rose-200",
  lime: "bg-lime-100 border-lime-300 hover:bg-lime-200",
  violet: "bg-violet-100 border-violet-300 hover:bg-violet-200",
};

export default function QuestionButton({
  text,
  color = "amber",
  selected,
  onSelect,
  setQuestion,
  setAnswer,
  setStage,
}: Props) {
  const [visible, setVisible] = useState(true);

  const handleClick = () => {
    onSelect(); // 선택 상태 설정
    setVisible(false); // 애니메이션으로 제거
  };

  useEffect(() => {
    if (!visible && selected) {
      const timer = setTimeout(() => {
        setQuestion(text);
        setAnswer("");
        setStage("answering");
        sendMessage(text);
      }, 400);
      return () => clearTimeout(timer);
    }
  }, [visible, selected]);

  return (
    <AnimatePresence>
      {visible && (
        <motion.button
          onClick={handleClick}
          initial={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.4 }}
          className={`relative w-full text-left p-6 mb-4 rounded-xl border shadow-md min-h-[120px] transition-all duration-200 font-[600] ${colorClassMap[color]}`}
        >
          {/* 핀 아이콘 */}
          <div className="absolute -top-2 left-4 text-xl z-20">📌</div>

          {/* 메모지 줄 배경 */}
          <div
            className="absolute inset-0 bg-repeat-y bg-[linear-gradient(to_bottom,transparent_28px,rgba(0,0,0,0.07)_29px)]"
            style={{ backgroundSize: "100% 32px", zIndex: 0 }}
          />

          {/* 질문 텍스트 */}
          <span className="relative z-10 text-gray-800 whitespace-pre-line text-[1.05rem] leading-relaxed font-['Nanum_Pen_Script','sans-serif']">
            {text}
          </span>
        </motion.button>
      )}
    </AnimatePresence>
  );
}
