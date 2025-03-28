package com.ssafy.happie.test.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/api/test")
public class TestController {
    @GetMapping("/test1")
    public ResponseEntity<?> test(){
        return ResponseEntity.ok("테스트 완료");
    }
}
