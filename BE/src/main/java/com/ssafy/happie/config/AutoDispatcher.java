package com.ssafy.happie.config;

import com.ssafy.happie.service.OrderService;
import lombok.RequiredArgsConstructor;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class AutoDispatcher {

    private final OrderService orderService;

    @Scheduled(fixedRate = 10000)
    public void autoSendDestination() {
        try {
            String result = orderService.sendDestination();
            System.out.println("AutoDispatcher: " + result);
        } catch (Exception e) {
            System.err.println("AutoDispatcher Error: " + e.getMessage());
        }
    }

    @Scheduled(fixedRate = 15000)
    public void autoSendOrder() {
        try {
            String result = orderService.autoDriving();
            System.out.println("AutoDispatcher: " + result);
        } catch (Exception e) {
            System.err.println("AutoDispatcher Error: " + e.getMessage());
        }
    }
}