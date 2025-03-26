"use client";

// 더미 데이터 (각 로봇의 카메라 이미지)
const dummyCameraFeeds: { [key: string]: string } = {
  1: "/images/dummy_camera_1.jpg", // 로봇 1의 더미 카메라 이미지
  2: "/images/dummy_camera_2.jpg", // 로봇 2의 더미 카메라 이미지
  3: "/images/dummy_camera_3.jpg", // 로봇 3의 더미 카메라 이미지
};

export default function BotCamera({ botId }: { botId: number }) {
  const imageSrc = dummyCameraFeeds[botId.toString()];

  return (
    <div className="w-full h-56 bg-gray-200 rounded-lg flex items-center justify-center shadow-md overflow-hidden">
      {imageSrc ? (
        <img
          src={imageSrc}
          alt={`로봇 ${botId} 카메라 화면`}
          className="w-full h-full object-cover"
        />
      ) : (
        <p className="text-gray-500">📷 로봇 {botId} 카메라 화면 없음</p>
      )}
    </div>
  );
}
