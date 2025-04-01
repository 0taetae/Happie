package com.ssafy.happie.service;

import com.ssafy.happie.dto.OrderRequestDto;
import com.ssafy.happie.dto.OrderResponseDto;
import com.ssafy.happie.entity.Order;
import com.ssafy.happie.repository.OrderRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class OrderService {
    private final OrderRepository orderRepository;

    @Transactional
    public OrderResponseDto createOrder(OrderRequestDto orderRequestDto) {
        Order order = new Order(orderRequestDto.getRobot(), orderRequestDto.getPlace(), orderRequestDto.getTodo());
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
}
