package com.ssafy.happie.repository;

import com.ssafy.happie.entity.Order;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface OrderRepository extends JpaRepository<Order, Integer> {
    List<Order> findByRobotAndState(String robot, String state);

    List<Order> findByRobotAndStateOrderByIdDesc(String robot, String state);

    Order findTop1ByRobotAndStateOrderByIdDesc(String robot, String state);

    Optional<Order> findFirstByRobotAndStateOrderByIdAsc(String robot, String state);

    boolean existsByRobotAndState(String robot, String state);
}
