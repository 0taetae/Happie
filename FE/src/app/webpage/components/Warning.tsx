"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { mqttClient } from "@/lib/mqttClient";

interface WarningProps {
  onClose: () => void;
  imageUrl: string;
}

// 원본 지도 해상도 (MQTT로 전달되는 좌표 기준)
const MAP_WIDTH = 1024;
const MAP_HEIGHT = 1024;

export default function Warning({ onClose, imageUrl }: WarningProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [mapImage, setMapImage] = useState<string | null>(null);
  const [robot1Position, setRobot1Position] = useState<{ x: number; y: number } | null>(null);

  // 🔈 경고음 재생
  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch((err) => console.warn("🔇 경고음 재생 실패:", err));
      }
    }

    return () => {
      if (audio) {
        audio.pause();
        audio.currentTime = 0;
      }
    };
  }, []);

  // 🛰️ MQTT 구독
  useEffect(() => {
    const handleMapMessage = (topic: string, message: Buffer) => {
      if (topic === "map/data") {
        try {
          const parsed = JSON.parse(message.toString());
          const base64 = parsed.image;
          setMapImage(`data:image/png;base64,${base64}`);
        } catch (err) {
          console.error("❌ 맵 데이터 파싱 오류:", err);
        }
      }

      if (topic === "robot/map_position") {
        const [x, y] = message.toString().split(",").map(Number);
        if (!isNaN(x) && !isNaN(y)) {
          setRobot1Position({ x, y });
        }
      }
    };

    mqttClient.subscribe("map/data");
    mqttClient.subscribe("robot/map_position");
    mqttClient.on("message", handleMapMessage);

    return () => {
      mqttClient.unsubscribe("map/data");
      mqttClient.unsubscribe("robot/map_position");
      mqttClient.off("message", handleMapMessage);
    };
  }, []);

  // 닫기 + MQTT 전송
  const handleClose = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }

    mqttClient.publish("robot/fall_check", "check");
    console.log("📤 MQTT 전송: robot/fall_check → check");
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 px-4 py-10">
      <audio ref={audioRef} src="/sounds/warning.mp3" loop autoPlay />

      <div className="bg-white border-4 border-red-500 rounded-2xl p-6 sm:p-10 md:p-12 w-full max-w-6xl shadow-2xl animate-fadeInModal">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-6xl text-center flex-grow">⚠️ 사고 발생 ⚠️</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* 낙상 이미지 */}
          <div className="border-4 border-yellow-400 rounded-xl p-2 flex items-center justify-center">
            <Image
              src={imageUrl || "/images/fall.png"}
              alt="넘어진 노인"
              width={600}
              height={600}
              unoptimized
              className="object-contain w-full h-auto rounded-lg max-h-[50vh]"
            />
          </div>

          {/* 지도 + 마커 */}
          <div className="border-4 border-yellow-400 rounded-xl p-2 relative flex items-center justify-center bg-gray-100">
            {mapImage ? (
              <img
                src={mapImage}
                alt="실시간 지도"
                className="object-contain w-full h-auto rounded-lg max-h-[50vh]"
              />
            ) : (
              <span className="text-gray-400">🕓 지도를 불러오는 중...</span>
            )}

            {robot1Position && (
              <div
                className="absolute z-20"
                style={{
                  left: `${(robot1Position.x / MAP_WIDTH) * 100}%`,
                  top: `${(robot1Position.y / MAP_HEIGHT) * 100}%`,
                  transform: "translate(-50%, -50%)",
                }}
              >
                <div className="w-5 h-5 bg-red-500 rounded-full border-2 border-white shadow-md" />
              </div>
            )}
          </div>
        </div>

        <div className="flex justify-center">
          <button
            onClick={handleClose}
            className="bg-red-500 text-white px-8 py-3 rounded-lg text-2xl hover:bg-red-600 transition"
          >
            확인했습니다
          </button>
        </div>
      </div>
    </div>
  );
}
