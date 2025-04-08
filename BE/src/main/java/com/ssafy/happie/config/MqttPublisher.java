package com.ssafy.happie.config;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.json.simple.JSONObject;

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

    public void sendEquipment(int id, String equip, double x, double y){
        System.out.println("id" + id + ", equip: "+ equip + ", x: "+ x + ", y: " + y);

        try{
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("id", id);
            jsonObject.put("equip", equip);
            jsonObject.put("x", x);
            jsonObject.put("y", y);

            MqttMessage message = new MqttMessage(jsonObject.toString().getBytes());
            message.setQos(1);

            mqttClient.publish("robot/move/equipment", message);
            System.out.println("메시지 전송 완료:" + jsonObject.toString());
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }

    public void sendNavComplete(String where, String status, String todo){
        try{
            System.out.println("where: " + where);
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("place", where);
            jsonObject.put("status", status);
            jsonObject.put("todo", todo);

            MqttMessage message = new MqttMessage(jsonObject.toString().getBytes());
            message.setQos(1);

            mqttClient.publish("robot/nav/complete", message);
            System.out.println("메시지 전송 완료:" + jsonObject.toString());
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }
}
