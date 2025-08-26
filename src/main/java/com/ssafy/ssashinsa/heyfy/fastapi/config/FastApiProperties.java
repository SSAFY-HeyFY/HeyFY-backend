package com.ssafy.ssashinsa.heyfy.fastapi.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "external.fastapi")
@Getter
@Setter
public class FastApiProperties {
    private String baseUrl;
    private String port;

    public String getFullBaseUrl() {
        return baseUrl + (port != null ? ":" + port : "8888");
    }
}