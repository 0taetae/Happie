package com.ssafy.happie.config;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MqttPublisherConfig {
    @Bean
    public MqttPublisher mqttPublisher(@Value("${mqtt.broker}") String broker,
                                       @Value("${mqtt.client-id}") String clientId,
                                       @Value("${mqtt.topic}") String topic,
                                       @Value("${mqtt.username}") String username,
                                       @Value("${mqtt.password}") String password) throws MqttException {
        return new MqttPublisher(broker, clientId, topic, username, password);
    }

    @Bean
    public MqttClient mqttClient(@Value("${mqtt.broker}") String broker,
                                 @Value("${mqtt.client-id-subscriber}") String subscriberId,
                                 @Value("${mqtt.username}") String username,
                                 @Value("${mqtt.password}") String password) throws MqttException {

        MqttClient mqttClient = new MqttClient(broker, subscriberId, new MemoryPersistence());

        // 옵션 설정
        MqttConnectOptions options = new MqttConnectOptions();
        options.setUserName(username);
        options.setPassword(password.toCharArray());
        options.setCleanSession(false);              // 세션 유지
        options.setAutomaticReconnect(true);         // 자동 재연결
        options.setKeepAliveInterval(60);            // 헬스 체크
        options.setConnectionTimeout(10);            // 연결 타임아웃

        mqttClient.connect(options);
        return mqttClient;
    }
}
