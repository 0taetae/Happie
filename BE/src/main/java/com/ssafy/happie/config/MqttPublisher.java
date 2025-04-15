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
        try {
            if (!mqttClient.isConnected()) {
                System.err.println("MQTT 연결 안됨! 메시지 전송 실패 (robot/destination)");
                 mqttClient.reconnect();
                return;
            }

            JSONObject jsonObject = new JSONObject();
            jsonObject.put("id", id);
            jsonObject.put("x", x);
            jsonObject.put("y", y);

            MqttMessage message = new MqttMessage(jsonObject.toString().getBytes());
            message.setQos(1);

            mqttClient.publish("robot/destination", message);
            System.out.println("📡 메시지 전송 완료: " + jsonObject.toString());
        } catch (MqttException e) {
            System.err.println("🔥 MQTT 전송 실패: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public void sendEquipment(int id, int equip, double x, double y){
        System.out.println("id" + id + ", no: "+ equip + ", x: "+ x + ", y: " + y);

        try {
//            if (!mqttClient.isConnected()) {
//                System.out.println("MQTT 연결이 끊겨있어 재연결 시도 중...");
//                mqttClient.reconnect(); // 또는 mqttClient.connect(options); 다시 설정해도 됨
//            }

            JSONObject jsonObject = new JSONObject();
            jsonObject.put("id", id);
            jsonObject.put("no", equip);
            jsonObject.put("x", x);
            jsonObject.put("y", y);

            MqttMessage message = new MqttMessage(jsonObject.toString().getBytes());
            message.setQos(1);

            mqttClient.publish("robot/equipment", message);
            System.out.println("메시지 전송 완료:" + jsonObject.toString());
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }


//    public void sendNavComplete(String where, String status, String todo){
//        try{
//            System.out.println("where: " + where);
//            JSONObject jsonObject = new JSONObject();
//            jsonObject.put("place", where);
//            jsonObject.put("status", status);
//            jsonObject.put("todo", todo);
//
//            MqttMessage message = new MqttMessage(jsonObject.toString().getBytes());
//            message.setQos(1);
//
//            mqttClient.publish("robot/nav/complete", message);
//            System.out.println("메시지 전송 완료:" + jsonObject.toString());
//        } catch (MqttException e) {
//            e.printStackTrace();
//        }
//    }

    public void autoDriving(int id, String status) {
        try {
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("id", id);
            jsonObject.put("status", status);

            MqttMessage message = new MqttMessage(jsonObject.toString().getBytes());
            message.setQos(1);

            mqttClient.publish("robot/patrol", message);
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }

    public void cleanEquipment(int id, int equip, String status) {
        try {
            JSONObject jsonObject = new JSONObject();

            jsonObject.put("id", id);
            jsonObject.put("no", equip);
            jsonObject.put("status", status);

            MqttMessage message = new MqttMessage(jsonObject.toString().getBytes());
            message.setQos(1);

            mqttClient.publish("robot/clean", message);
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }
}
