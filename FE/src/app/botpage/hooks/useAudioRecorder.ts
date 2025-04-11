// hooks/useAudioRecorder.ts
import { useRef } from "react";

export function useAudioRecorder(onRecordingComplete: (blob: Blob) => void) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        onRecordingComplete(audioBlob); // 전송은 외부에서!
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      console.log("🎤 녹음 시작");

      setTimeout(() => {
        mediaRecorder.stop();
      }, 5000);
    } catch (err) {
      console.error("❌ 마이크 접근 실패:", err);
    }
  };

  return { startRecording };
}
