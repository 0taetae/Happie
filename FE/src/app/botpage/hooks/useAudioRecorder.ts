import { useRef } from "react";
import { mqttClient } from "../../../lib/mqttClient";
//녹음만 담당하는 훅
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
        console.log("🛑 녹음 종료");
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      
        // MQTT 전송
        audioBlob.arrayBuffer().then((buffer) => {
          const base64Data = Buffer.from(buffer).toString("base64");
          mqttClient.publish("my/topic", base64Data);
          console.log("📤 음성 데이터 MQTT 전송 완료!");
        });
      
        onRecordingComplete(audioBlob);
      };
      

      mediaRecorderRef.current = mediaRecorder;

      console.log("🎤 녹음 시작!");
      mediaRecorder.start();

      setTimeout(() => {
        mediaRecorder.stop();
      }, 5000);
    } catch (err) {
      console.error("❌ 마이크 접근 실패:", err);
    }
  };

  return { startRecording };
}
