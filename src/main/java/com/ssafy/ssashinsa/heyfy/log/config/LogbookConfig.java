package com.ssafy.ssashinsa.heyfy.log.config;

import java.util.Set;
import java.util.HashSet;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.zalando.logbook.Logbook;

@Configuration
public class LogbookConfig {

    @Bean
    public Logbook logbook() {
        Set<String> sensitiveKeys = new HashSet<>();
        sensitiveKeys.add("password");
        sensitiveKeys.add("accessToken");
        sensitiveKeys.add("refreshToken");
        sensitiveKeys.add("authorization");
        sensitiveKeys.add("rrn");
        sensitiveKeys.add("phone");
        sensitiveKeys.add("cardNumber");
        sensitiveKeys.add("accountNumber");

        return Logbook.builder()
                .bodyFilter(new CustomBodyFilter(sensitiveKeys))
                .build();
    }
}