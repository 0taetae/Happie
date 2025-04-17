package com.ssafy.happie.service;

import com.ssafy.happie.config.MqttPublisher;
import com.ssafy.happie.dto.OrderRequestDto;
import com.ssafy.happie.dto.OrderResponseDto;
import com.ssafy.happie.entity.Order;
import com.ssafy.happie.repository.OrderRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class OrderService {
    private final OrderRepository orderRepository;

    private final MqttPublisher mqttPublisher;

    // 맵 좌표
    private static final Map<String, double[]> PLACE_COORDINATES = Map.of(
        "501호실", new double[] {-52.60654067993164, -38.33089828491211},
        "502호실", new double[] {-47.468528747558594, -38.27075958251953},
        "503호실", new double[] {-42.64277267456055, -38.69271469116211},
        "간호사실", new double[] {-50.82450485229492, -54.83995056152344},
        "휠체어 보관실", new double[] {-56.482933044433594, -52.840431213378906},
        "링거 보관실", new double[] {-52.648616790771484 , -38.33458709716797},
        "로봇방", new double[] {-42.52598571777344, -46.45439147949219}
    );


    @Transactional
    public OrderResponseDto createOrder(OrderRequestDto orderRequestDto) {
        String place = orderRequestDto.getPlace();

        // 장소에 따른 좌표를 가져옴
        double[] coordinates = PLACE_COORDINATES.getOrDefault(place, new double[]{0, 0}); // 기본값은 0,0
        System.out.println(orderRequestDto.getTodo());
        Order order = new Order(
                orderRequestDto.getRobot(),
                place,
                orderRequestDto.getTodo(),
                coordinates[0], // x 좌표
                coordinates[1]  // y 좌표
        );

        orderRepository.save(order);

        String result = sendDestination();
        System.out.println("MQTT 전송 결과: " + result);

        return OrderResponseDto.builder()
                .Id(order.getId())
                .robot(order.getRobot())
                .place(order.getPlace())
                .todo(order.getTodo())
                .state(order.getState())
                .build();
    }

    @Transactional(readOnly = true)
    public List<OrderResponseDto> commamdList(String robot) {
        List<Order> orderlist = orderRepository.findByRobotAndState(robot, "대기");

        return orderlist.stream()
                .map(order -> OrderResponseDto.builder()
                        .Id(order.getId())
                        .robot(order.getRobot())
                        .place(order.getPlace())
                        .todo(order.getTodo())
                        .state(order.getState())
                        .build())
                .collect(Collectors.toList());
    }

    @Transactional
    public String deleteOrder(int orderId) {
        orderRepository.findById(orderId).orElseThrow(
                () -> new RuntimeException("OrderId not found")
        );
        orderRepository.deleteById(orderId);

        return "명령 삭제 완료";
    }

    @Transactional(readOnly = true)
    public List<OrderResponseDto> completeList(String robot) {
        List<Order> completelist = orderRepository.findByRobotAndStateOrderByIdDesc(robot, "완료");

        return completelist.stream()
                .map(order -> OrderResponseDto.builder()
                        .Id(order.getId())
                        .robot(order.getRobot())
                        .place(order.getPlace())
                        .todo(order.getTodo())
                        .state(order.getState())
                        .build())
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public OrderResponseDto inProgressOrder(String robot) {
        Order order = orderRepository.findTop1ByRobotAndStateOrderByIdDesc(robot, "진행 중");

        return OrderResponseDto.builder()
                .Id(order.getId())
                .robot(order.getRobot())
                .place(order.getPlace())
                .todo(order.getTodo())
                .state(order.getState())
                .build();
    }

    @Transactional
    public String sendDestination() {
        String robot = "robot1";

        boolean inProgressExists = orderRepository.existsByRobotAndState(robot, "진행 중");

        if (inProgressExists) {
            return "진행 중인 명령이 있어 전송할 수 없습니다.";
        }

        Order order = orderRepository.findFirstByRobotAndStateOrderByIdAsc(robot, "대기")
                .orElseThrow(() -> new IllegalArgumentException("robot1의 대기 중인 명령이 없습니다."));

        String todo = order.getTodo();
        order.setState("진행 중");
        orderRepository.save(order);

        if (todo.equals("운행")) {
            mqttPublisher.autoDriving(order.getId(), "start");

            return String.format("자율주행 명령 전송 완료 (id = %d)", order.getId());
        } else if (todo.contains("정리")) {
            String item = todo.split(" ")[0];
            int type = item.equals("휠체어") ? 1 : 2;

            mqttPublisher.cleanEquipment(order.getId(), type, "start");

            return String.format("정리 명령 전송 완료 (id = %d)", order.getId());
        } else if (todo.contains("전달")) {
            String item = todo.split(" ")[0];  // "휠체어" 또는 "링거"
            int type = item.equals("휠체어") ? 1 : 2;

            double x = order.getX();
            double y = order.getY();

            mqttPublisher.sendEquipment(order.getId(), type, x, y);

            return String.format("MQTT 전송 완료 (기자재 전달) id = %d, type = %d, x = %.6f, y = %.6f", order.getId(), type, x, y);
        } else if (todo.equals("안내")) {
            try {
                mqttPublisher.sendLocation(order.getId(), order.getX(), order.getY());
            } catch (Exception e) {
                System.err.println("❌ 안내 명령 MQTT 전송 실패: " + e.getMessage());
                order.setState("대기");
                orderRepository.save(order);
                throw new RuntimeException("MQTT 전송 실패 - 안내 명령 롤백");
            }

            return String.format("안내 명령 MQTT 전송 완료 (id = %d)", order.getId());
        }
        // 그 외 일반 위치 이동 명령 처리
        mqttPublisher.sendLocation(order.getId(), order.getX(), order.getY());

        return String.format("MQTT 전송 완료 id = %d, x = %.6f, y = %.6f", order.getId(), order.getX(), order.getY());
    }

    @Transactional
    public void robotLog(int id, String status) {
        if (id == -2) {
            System.out.println("잘못된 ID: -2");
            return;
        }
        Order order = orderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("해당 ID의 명령이 없습니다: " + id));

//        String todo = order.getTodo();

//        if (status.equals("arrive")) {
//            if (!todo.contains("전달")) {
//                System.out.println("전달 명령이 아니므로 좌표 재전송 생략");
//                return;
//            }
//
//            String item = todo.split(" ")[0];  // "휠체어" 또는 "링거"
//            int type = item.equals("휠체어") ? 1 : 2;
//
//            // 좌표 재전송
//            mqttPublisher.sendEquipment(order.getId(), type, order.getX(), order.getY());
//            System.out.printf("📦 좌표 재전송 완료 (id = %d, type = %d, x = %.6f, y = %.6f)%n",
//                    order.getId(), type, order.getX(), order.getY());

//        }
//        else if (status.equals("finish")) {
        if (status.equals("finish")) {
            order.setState("완료");
            orderRepository.save(order);
            System.out.printf("명령 상태 완료 처리됨 (id = %d)%n", order.getId());

        } else {
            System.out.printf("처리되지 않은 상태: %s%n", status);
        }
    }

    @Transactional
    public String autoDriving() {
        String robot = "robot1";

        boolean notFinishedExists = orderRepository.existsByRobotAndStateNot(robot, "완료");

        if (notFinishedExists) {
            return "완료되지 않은 명령 존재";
        }

        mqttPublisher.autoDriving(-1, "start");
        return "전체 순회 명령 전송 완료";
    }
}
