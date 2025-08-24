package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.EntireExchangeRateResponseDto;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;

@SpringBootTest
class ShinhanExchangeRateApiClientTest {

    @Autowired
    ShinhanExchangeRateApiClient client;

    @TestConfiguration
    static class WebClientTestConfig {
        @Bean
        public WebClient.Builder webClientBuilder() {
            return WebClient.builder();
        }
    }

    @Test
    @DisplayName("환율 조회 API 실행 테스트")
    void getExchangeRateFromExternalApiTest() {
        try {
            EntireExchangeRateResponseDto result = client.getExchangeRateFromExternalApi();
        } catch (Exception e) {
            System.out.println("getExchangeRateFromExternalApi error: " + e.getMessage());
        }
    }
}

