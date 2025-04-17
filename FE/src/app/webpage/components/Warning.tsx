"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import { mqttClient } from "@/lib/mqttClient";
import { getMapImageData, getMapParamsData } from "@/lib/mapStore";

interface WarningProps {
  onClose: () => void;
  imageUrl: string;
}

export default function Warning({ onClose, imageUrl }: WarningProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [mapImage, setMapImage] = useState<string | null>(null);
  const [mapParams, setMapParams] = useState<{ MAP_SIZE: number[]; MAP_RESOLUTION: number } | null>(null);
  const [robot1Position, setRobot1Position] = useState<{ x: number; y: number } | null>(null);

  // 🧠 계산된 지도 해상도 (픽셀 단위)
  const mapWidthPx = useMemo(() => {
    return mapParams ? mapParams.MAP_SIZE[0] / mapParams.MAP_RESOLUTION : 1024;
  }, [mapParams]);

  const mapHeightPx = useMemo(() => {
    return mapParams ? mapParams.MAP_SIZE[1] / mapParams.MAP_RESOLUTION : 1024;
  }, [mapParams]);

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

  // 🗺️ 전역 상태에서 이미지 및 해상도 가져오기
  useEffect(() => {
    const savedImage = getMapImageData();
    const savedParams = getMapParamsData();
    if (savedImage) setMapImage(savedImage);
    if (savedParams) setMapParams(savedParams);
  }, []);

  // 🛰️ MQTT 수신 처리
  useEffect(() => {
    const handleMapMessage = (topic: string, message: Buffer) => {
      if (topic === "robot/map_position") {
        const [x, y] = message.toString().split(",").map(Number);
        if (!isNaN(x) && !isNaN(y)) {
          setRobot1Position({ x, y });
        }
      }
    };

    mqttClient.subscribe("robot/map_position");
    mqttClient.on("message", handleMapMessage);

    return () => {
      mqttClient.unsubscribe("robot/map_position");
      mqttClient.off("message", handleMapMessage);
    };
  }, []);

  // 닫기 + MQTT 발신
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
        <h2 className="text-6xl text-center mb-10">⚠️ 사고 발생 ⚠️</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* 🔸 낙상 이미지 */}
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

          {/* 🔸 지도 + 마커 */}
          <div className="border-4 border-yellow-400 rounded-xl p-2 relative w-full aspect-square bg-gray-100 overflow-hidden">
            {mapImage ? (
              <img
                src={mapImage}
                alt="실시간 지도"
                className="absolute inset-0 object-contain w-full h-full rounded-lg"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-2xl">
                🕓 지도를 불러오는 중...
              </div>
            )}

            {/* ✅ 실시간 마커 위치 */}
            {robot1Position && (
              <div
                className="absolute z-20"
                style={{
                  left: `${(robot1Position.x / mapWidthPx) * 100}%`,
                  top: `${(robot1Position.y / mapHeightPx) * 100}%`,
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
