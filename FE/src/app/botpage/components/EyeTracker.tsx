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
        const maxDistance = eyeRect.width * 0.25;

        const moveX = Math.cos(angle) * maxDistance;
        const moveY = Math.sin(angle) * maxDistance;

        pupil.style.transform = `translate(-50%, -50%) translate(${moveX}px, ${moveY}px)`;
      };

      movePupil(leftEyeRef.current);
      movePupil(rightEyeRef.current);
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  useEffect(() => {
    const pupils = document.querySelectorAll(".pupil");

    const growAndShrink = () => {
      pupils.forEach((p) => {
        const el = p as HTMLElement;
        el.style.transition = "transform 0.3s ease";
        el.style.transform += " scale(1.2)";
      });

      setTimeout(() => {
        pupils.forEach((p) => {
          const el = p as HTMLElement;
          el.style.transform = el.style.transform.replace(" scale(1.2)", "");
        });
      }, 3000); // 3초 후 원래대로
    };

    const interval = setInterval(growAndShrink, 10000); // 10초마다
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const eyelids = document.querySelectorAll(".eyelid");

    const blink = () => {
      eyelids.forEach((lid) => (lid as HTMLElement).style.height = "100%");
      setTimeout(() => {
        eyelids.forEach((lid) => (lid as HTMLElement).style.height = "0%");
      }, 150);
    };

    const blinkInterval = setInterval(blink, 5000);
    return () => clearInterval(blinkInterval);
  }, []);

  return (
    <div className="flex justify-center gap-20 mb-12 py-6">
      {[leftEyeRef, rightEyeRef].map((ref, idx) => (
        <div
          key={idx}
          ref={ref}
          className="w-48 h-48 bg-white rounded-full shadow-inner border-2 border-gray-300 relative overflow-hidden"
        >
          <div
            className="pupil w-24 h-24 bg-black rounded-full absolute top-1/2 left-1/2"
            style={{
              transform: "translate(-50%, -50%)",
              transition: "transform 0.15s ease",
            }}
          />
          <div
            className="eyelid absolute top-0 left-0 w-full h-0 bg-white z-10"
            style={{
              transition: "height 0.2s ease-in-out",
            }}
          />
        </div>
      ))}
    </div>
  );
}
