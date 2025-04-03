package com.ssafy.happie.config;

import org.eclipse.paho.client.mqttv3.MqttException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MqttPublisherConfig {

    @Bean
    public MqttPublisher mqttPublisher(@Value("${mqtt.broker}") String broker,
                                       @Value("${mqtt.client-id}") String clientId,
                                       @Value("${mqtt.topic}") String topic) throws MqttException {
        return new MqttPublisher(broker, clientId, topic);
    }
}
