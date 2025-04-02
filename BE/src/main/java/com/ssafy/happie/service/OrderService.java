package com.ssafy.happie.service;

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

    // 맵 좌표
    private static final Map<String, String[]> PLACE_COORDINATES = Map.of(
            "병실 1", new String[] {"36.58", "-52.52"},
            "병실 2", new String[] {"36.65", "-47.51"},
            "병실 3", new String[] {"36.55", "-42.56"},
            "데스크", new String[] {"55.37", "-50.85"},
            "휠체어 보관", new String[] {"53.21", "-56.75"},
            "링거 보관", new String[] {"53.38", "-60.21"},
            "로봇방", new String[] {"44.93", "-42.44"}
    );

    @Transactional
    public OrderResponseDto createOrder(OrderRequestDto orderRequestDto) {
        String place = orderRequestDto.getPlace();
        String[] coordinates = PLACE_COORDINATES.getOrDefault(place, new String[]{"0", "0"}); // 기본값은 0,0

        Order order = new Order(
                orderRequestDto.getRobot(),
                place,
                orderRequestDto.getTodo(),
                coordinates[0], // x
                coordinates[1]  // y
        );

        orderRepository.save(order);

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
}
