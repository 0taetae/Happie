package com.ssafy.happie.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.happie.service.OrderService;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.eclipse.paho.client.mqttv3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class MqttSubscriber {

    private final MqttClient mqttClient;
    private final OrderService orderService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${mqtt.topic.complete}")
    private String topic;

    @PostConstruct
    public void subscribeToLogTopic() {
        System.out.println("✅ MqttSubscriber 초기화 시작");

        try {
            mqttClient.subscribe("robot/log", (topic, message) -> {
                String payload = new String(message.getPayload());
                System.out.println("📩 수신한 메시지: " + payload); // <- 반드시 보이도록!

                try {
                    JsonNode json = objectMapper.readTree(payload);
                    int id = json.get("id").asInt();
                    String status = json.get("status").asText();

                    System.out.printf("🧪 파싱된 id: %d, status: %s%n", id, status);

                    orderService.robotLog(id, status);

                } catch (Exception e) {
                    System.err.println("❌ JSON 파싱 오류: " + e.getMessage());
                }
            });

            System.out.println("✅ robot/log 구독 완료");

        } catch (MqttException e) {
            System.err.println("❌ MQTT 구독 실패: " + e.getMessage());
        }
    }
}