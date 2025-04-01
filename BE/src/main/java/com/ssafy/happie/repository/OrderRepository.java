package com.ssafy.happie.repository;

import com.ssafy.happie.entity.Order;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface OrderRepository extends JpaRepository<Order, Integer> {
    List<Order> findByRobotAndState(String robot, String state);
}
