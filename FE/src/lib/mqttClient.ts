import mqtt from "mqtt";

const brokerUrl = "wss://j12e103.p.ssafy.io/ws/";

const options = {

  clientId: `nextjs_mqtt_${Math.random().toString(16).substr(2, 8)}`,
  reconnectPeriod: 10000000,
  clean: true,
};

export const mqttClient = mqtt.connect(brokerUrl, options);

mqttClient.on("connect", () => {
  console.log("✅ MQTT 연결 성공");

  // ✅ 필요한 모든 토픽 구독
  const topics = [
    "user/chatbot/request",
    "fall_detection",
    "chatbot/response", // 🔹 LLM 응답 수신을 위한 구독
  ];

  topics.forEach((topic) => {
    mqttClient.subscribe(topic, (err) => {
      if (err) console.error(`❌ 구독 실패: ${topic}`, err);
      else console.log(`✅ 구독 성공: ${topic}`);
    });
  });
});

mqttClient.on("error", (err) => {
  console.error("❌ MQTT 오류:", err);
});
