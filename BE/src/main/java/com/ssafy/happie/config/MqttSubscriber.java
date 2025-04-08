package com.ssafy.happie.config;

import com.ssafy.happie.service.OrderService;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.eclipse.paho.client.mqttv3.*;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class MqttSubscriber {
    private final OrderService orderService;

    @Value("${mqtt.broker}")
    private String broker;

    @Value("${mqtt.client-id-subscriber}")
    private String clientId;

    @Value("${mqtt.username}")
    private String username;

    @Value("${mqtt.password}")
    private String password;

    @Value("${mqtt.topic.destination-request}")
    private String destinationTopic;

    @Value("${mqtt.topic.complete}")
    private String completeTopic;
}
