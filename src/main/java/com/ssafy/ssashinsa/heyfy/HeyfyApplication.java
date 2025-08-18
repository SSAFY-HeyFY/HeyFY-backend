package com.ssafy.ssashinsa.heyfy;

import com.ssafy.ssashinsa.heyfy.transfer.config.SsafyFinApiProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties(SsafyFinApiProperties.class)
public class HeyfyApplication {

	public static void main(String[] args) {
		SpringApplication.run(HeyfyApplication.class, args);
	}

}
