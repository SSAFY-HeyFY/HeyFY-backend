package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanExchangeResponseDto;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;

@SpringBootTest
class ShinhanExchangeApiClientTest {

    @Autowired
    ShinhanExchangeApiClient client;

    @TestConfiguration
    static class WebClientTestConfig {
        @Bean
        public WebClient.Builder webClientBuilder() {
            return WebClient.builder();
        }
    }

    @Test
    @DisplayName("환전 실행 API 테스트")
    void exchangeTest() {
        try {
            // 예시 값: 실제 테스트 환경에 맞게 수정 필요
            String accountNo = "0017459003466586";
            String exchangeCurrency = "USD";
            Double exchangeAmount = 100.0;
            String userKey = "e6ab7c69-db18-414e-b9a0-2fedf94eaf7f";
            ShinhanExchangeResponseDto result = client.exchange(accountNo, exchangeCurrency, exchangeAmount, userKey);
            System.out.println("exchange result: " + result);
        } catch (Exception e) {
            System.out.println("exchange error: " + e.getMessage());
        }
    }
}

