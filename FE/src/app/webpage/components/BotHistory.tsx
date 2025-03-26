"use client";

const dummyHistoryData: Record<number, { id: number; log: string; timestamp: string }[]> = {
  1: [
    { id: 1, log: "약제실에서 약품 수령 완료", timestamp: "오후 3:30:00" },
    { id: 2, log: "병실 2로 이동 완료", timestamp: "오후 3:35:15" },
    { id: 3, log: "환자에게 링거 전달 완료", timestamp: "오후 3:40:20" },
    { id: 4, log: "병실 5로 휠체어 이동 완료", timestamp: "오후 3:45:10" },
    { id: 5, log: "병실 7로 이동 완료", timestamp: "오후 3:50:30" },
    { id: 6, log: "응급실에서 약품 전달 완료", timestamp: "오후 3:55:45" },
    { id: 7, log: "응급실에서 약품 전달 완료", timestamp: "오후 3:55:45" },
    { id: 8, log: "응급실에서 약품 전달 완ㅇㄹㄴㅁㅇㄻㅇㄻㄴㅇ료", timestamp: "오후 3:55:45" },
    { id: 9, log: "응급실에서 약품 전달 완료", timestamp: "오후 3:55:45" },
    { id: 10, log: "응급실에서 약품 전달 완료", timestamp: "오후 3:55:45" },
  ],
  2: [
    { id: 1, log: "응급실에서 의료 장비 배송 완료", timestamp: "오후 3:15:00" },
    { id: 2, log: "병실 3으로 이동 완료", timestamp: "오후 3:25:40" },
  ],
  3: [
    { id: 1, log: "충전 스테이션에서 충전 시작", timestamp: "오후 3:00:00" },
    { id: 2, log: "병동 3층 순찰 완료", timestamp: "오후 3:10:30" },
  ],
};

export default function BotHistory({ botId }: { botId: number }) {
  const history = dummyHistoryData[botId] || [];

  return (
    <div className="flex flex-col w-full bg-white rounded-lg shadow-md h-full">
      {/* ✅ BotHistory 높이 확장 및 개별 스크롤 적용 */}
      <div className="flex-grow px-1 max-h-[300px]">
        <ul className="space-y-2">
          {history.length > 0 ? (
            history
            .slice() // 원본 배열 유지
            .reverse() // 최신순 정렬
            .map((entry) => (
              <li key={entry.id} className="bg-gray-100 p-3 rounded flex justify-between shadow-sm">
                <span className="flex-1 break-words pr-4">🔹 {entry.log}</span>
                <span className="text-xs text-gray-500 whitespace-nowrap flex-shrink-0">{entry.timestamp}</span>
              </li>
            ))
          ) : (
            <li className="text-gray-500 text-sm">⏳ 활동 내역이 없습니다.</li>
          )}
        </ul>
      </div>
    </div>
  );
}
