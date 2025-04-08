package com.ssafy.happie.config;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;

public class MqttPublisher {
    private final MqttClient mqttClient;
    private final String topic;

    public MqttPublisher(String broker, String clientId, String topic, String username, String password) throws MqttException {
        this.topic = topic;
        this.mqttClient = new MqttClient(broker, clientId, new MemoryPersistence());

        MqttConnectOptions options = new MqttConnectOptions();
        options.setAutomaticReconnect(true);
        options.setCleanSession(true);
        options.setUserName(username);
        options.setPassword(password.toCharArray());

        mqttClient.connect(options);
    }

    public void sendLocation(int id, double x, double y) {
        String payload = String.format("{\"id\": %d, \"x\": %.6f, \"y\": %.6f}", id, x, y);

        try {
            MqttMessage message = new MqttMessage(payload.getBytes());
            message.setQos(1);
            mqttClient.publish(topic, message);
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }
}
