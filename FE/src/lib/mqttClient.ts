// src/lib/mqttClient.ts
import mqtt from 'mqtt';

const brokerUrl = 'ws://j12e103.p.ssafy.io:8083';

const options = {
  username: 'happie_mqtt_user',
  password: 'gkstkfckdl0411!',
  clientId: `nextjs_mqtt_${Math.random().toString(16).substr(2, 8)}`,
  reconnectPeriod: 1000,
  clean: true,
};

export const mqttClient = mqtt.connect(brokerUrl, options);

mqttClient.on('connect', () => {
  console.log('✅ MQTT 연결 성공');
  mqttClient.subscribe('my/topic', (err) => {
    if (err) console.error('❌ 구독 실패:', err);
    else console.log('✅ 구독 성공: my/topic');
  });
});

mqttClient.on('message', (topic, message) => {
  console.log(`📩 수신 [${topic}]: ${message.toString()}`);
});

mqttClient.on('error', (err) => {
  console.error('❌ MQTT 오류:', err);
});
