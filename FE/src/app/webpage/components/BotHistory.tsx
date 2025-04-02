"use client";

import { useEffect, useState } from "react";

interface HistoryEntry {
  id: number;
  robot: string;
  place: string;
  todo: string;
  state: string;
}

export default function BotHistory({ botId }: { botId: number }) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      setLoading(true);
      try {
        const robotKey = `robot${botId}`;
        const response = await fetch(
          `https://j12e103.p.ssafy.io/api/equipment/orders-reverse/${robotKey}`
        );
        if (!response.ok) throw new Error("서버 응답 오류");

        const data = await response.json();
        setHistory(data); // 데이터는 최신순으로 정렬된 상태라고 가정
      } catch (error) {
        console.error("활동 내역 불러오기 실패:", error);
        setHistory([]);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [botId]);

  return (
    <div className="flex flex-col w-full bg-white rounded-lg shadow-md h-full">
      <div className="flex-grow px-1 max-h-[300px] overflow-y-auto">
        <ul className="space-y-2">
          {loading ? (
            <li className="text-gray-400 text-sm text-center py-6">⌛ 불러오는 중...</li>
          ) : history.length > 0 ? (
            history.map((entry) => (
              <li
                key={entry.id}
                className="bg-gray-100 p-3 rounded flex justify-between shadow-sm"
              >
                <span className="flex-1 break-words pr-4">
                  🔹 {entry.place}에 {entry.todo} 완료
                </span>
              </li>
            ))
          ) : (
            <li className="text-gray-500 text-sm text-center">🫡 오늘은 꿈쩍도 안했어요.👍</li>
          )}
        </ul>
      </div>
    </div>
  );
}
