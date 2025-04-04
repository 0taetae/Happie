"use client";
import { useEffect, useRef } from "react";

export default function EyeTracker() {
  const leftEyeRef = useRef<HTMLDivElement>(null);
  const rightEyeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const movePupil = (eye: HTMLDivElement | null) => {
        if (!eye) return;

        const pupil = eye.querySelector(".pupil") as HTMLDivElement;
        const eyeRect = eye.getBoundingClientRect();

        const eyeCenterX = eyeRect.left + eyeRect.width / 2;
        const eyeCenterY = eyeRect.top + eyeRect.height / 2;

        const dx = e.clientX - eyeCenterX;
        const dy = e.clientY - eyeCenterY;

        const angle = Math.atan2(dy, dx);
        const maxDistance = eyeRect.width * 0.25; // 이동 최대치 = 눈 크기의 25%

        const moveX = Math.cos(angle) * maxDistance;
        const moveY = Math.sin(angle) * maxDistance;

        // 🔥 핵심: translate(-50%, -50%)는 유지하고, 추가 이동은 style로 처리
        pupil.style.setProperty("transform", `translate(-50%, -50%) translate(${moveX}px, ${moveY}px)`);
      };

      movePupil(leftEyeRef.current);
      movePupil(rightEyeRef.current);
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  return (
    <div className="flex justify-center gap-20 mb-12 py-6">
      {/* 왼쪽 눈 */}
      <div
        ref={leftEyeRef}
        className="w-48 h-48 bg-white rounded-full shadow-inner border-2 border-gray-300 relative overflow-hidden"
      >
        <div
          className="pupil w-24 h-24 bg-black rounded-full absolute top-1/2 left-1/2"
          style={{
            transform: "translate(-50%, -50%)", // 초기 중심 위치
            transition: "transform 0.15s ease",
          }}
        />
      </div>

      {/* 오른쪽 눈 */}
      <div
        ref={rightEyeRef}
        className="w-48 h-48 bg-white rounded-full shadow-inner border-2 border-gray-300 relative overflow-hidden"
      >
        <div
          className="pupil w-24 h-24 bg-black rounded-full absolute top-1/2 left-1/2"
          style={{
            transform: "translate(-50%, -50%)", // 초기 중심 위치
            transition: "transform 0.15s ease",
          }}
        />
      </div>
    </div>
  );
}
