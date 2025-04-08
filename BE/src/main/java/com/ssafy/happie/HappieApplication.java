package com.ssafy.happie;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableScheduling
@SpringBootApplication
public class HappieApplication {

	public static void main(String[] args) {
		SpringApplication.run(HappieApplication.class, args);
	}

}
