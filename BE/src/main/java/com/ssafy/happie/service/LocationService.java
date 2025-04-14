package com.ssafy.happie.service;

import com.ssafy.happie.config.MqttPublisher;
import com.ssafy.happie.dto.LocationRequestDto;
import com.ssafy.happie.dto.LocationResponseDto;
import com.ssafy.happie.entity.Order;
import com.ssafy.happie.repository.LocationRepository;
import com.ssafy.happie.repository.OrderRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class LocationService {
    private final LocationRepository locationRepository;

    private final OrderRepository orderRepository;

    private final MqttPublisher mqttPublisher;

    private final OrderService orderService;

    @Transactional
    public LocationResponseDto getLocationCoordinates(LocationRequestDto locationRequestDto) {
        String name = locationRequestDto.getName();

        return locationRepository.findByName(name)
                .map(location -> {
                    double x = location.getX();
                    double y = location.getY();

                    Order order = new Order(
                            "robot1",
                            name,
                            "안내",
                            x,
                            y
                    );
                    orderRepository.save(order);

                    orderService.sendDestination();

                    return new LocationResponseDto(order.getId(), x, y);
                })
                .orElseThrow(() -> new IllegalArgumentException("해당 시설명을 찾을 수 없습니다: " + name));
    }
}
