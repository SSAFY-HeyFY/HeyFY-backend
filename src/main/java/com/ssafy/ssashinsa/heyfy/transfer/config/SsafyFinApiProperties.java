package com.ssafy.ssashinsa.heyfy.transfer.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "ssafy.fin-api")
public record SsafyFinApiProperties(
        String baseUrl,
        String apiKey
) {}