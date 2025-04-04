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
    private static final Map<String, double[]> PLACE_COORDINATES = Map.of(
        "501호실", new double[] {-52.60654067993164, -38.33089828491211},
        "502호실", new double[] {-47.468528747558594, -38.27075958251953},
        "503호실", new double[] {-42.64277267456055, -38.69271469116211},
        "간호사실", new double[] {-50.82450485229492, -54.83995056152344},
        "휠체어 보관실", new double[] {-56.482933044433594, -52.840431213378906},
        "링거폴대 보관실", new double[] {-59.81382751464844, -52.48174285888672},
        "로봇방", new double[] {-42.52598571777344, -46.45439147949219}
    );


    @Transactional
    public OrderResponseDto createOrder(OrderRequestDto orderRequestDto) {
        String place = orderRequestDto.getPlace();

        // 장소에 따른 좌표를 가져옴
        double[] coordinates = PLACE_COORDINATES.getOrDefault(place, new double[]{0, 0}); // 기본값은 0,0

        Order order = new Order(
                orderRequestDto.getRobot(),
                place,
                orderRequestDto.getTodo(),
                coordinates[0], // x 좌표
                coordinates[1]  // y 좌표
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
