package com.ssafy.happie.controller;

import com.ssafy.happie.dto.OrderRequestDto;
import com.ssafy.happie.dto.OrderResponseDto;
import com.ssafy.happie.service.OrderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "equipment")
@Slf4j
@RestController
@RequestMapping("/api/equipment")
@RequiredArgsConstructor
public class OrderController {
    private final OrderService orderService;

    @PostMapping("/create-order")
    @Operation(summary = "명령 등록 API", description = "robot = 로봇1, place = 101호, todo = 휠체어 가져오기")
    public OrderResponseDto createOrder(@RequestBody OrderRequestDto orderRequestDto) {
        return orderService.createOrder(orderRequestDto);
    }

    @GetMapping("/orders/{robot}")
    @Operation(summary = "명령 조회 API", description = "로봇1, 로봇2, 로봇3의 대기중인 명령만 조회")
    public List<OrderResponseDto> commamdList(@PathVariable String robot) {
        return orderService.commamdList(robot);
    }

    @DeleteMapping("/order/{orderId}")
    @Operation(summary = "명령 삭제 API", description = "orderId")
    public String deleteOrder(@PathVariable int orderId) {
        return orderService.deleteOrder(orderId);
    }

//    @GetMapping
}
