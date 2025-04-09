"use client";

import { useEffect, useState } from "react";
import { mqttClient } from "@/lib/mqttClient";

// 더미 데이터
const CameraFeeds: { [key: string]: string } = {
  1: "/images/cameraloading.gif",
  2: "/images/charge.gif",
  3: "/images/repair.gif",
};

export default function BotCamera({ botId }: { botId: number }) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);

  useEffect(() => {
    if (botId !== 1) return;
  
    const topic = "robot/image";
  
    const handleMessage = (receivedTopic: string, message: Buffer) => {
      if (receivedTopic !== topic) return; // 💥 필수 조건
  
      const base64Image = message.toString();
  
      // base64 형식 아닌 좌표면 무시
      if (base64Image.includes(",") || base64Image.length < 100) {
        console.warn("❌ 잘못된 이미지 데이터 수신:", base64Image);
        return;
      }
  
      setImageSrc(`data:image/jpeg;base64,${base64Image}`);
    };
  
    mqttClient.subscribe(topic);
    mqttClient.on("message", handleMessage);
  
    return () => {
      mqttClient.unsubscribe(topic);
      mqttClient.off("message", handleMessage);
    };
  }, [botId]);
  
  

  const fallbackImage = CameraFeeds[botId.toString()];

  return (
    <div className="w-full h-56 rounded-lg flex items-center justify-center shadow-md overflow-hidden">
      {botId === 1 && imageSrc ? (
        <img
          src={imageSrc}
          alt={`로봇 1 카메라 화면`}
          className="w-full h-full object-cover"
        />
      ) : (
        <img
          src={fallbackImage}
          alt={`로봇 ${botId} 카메라`}
          className="w-full h-full object-contain"
        />

      )}
    </div>
  );
}
