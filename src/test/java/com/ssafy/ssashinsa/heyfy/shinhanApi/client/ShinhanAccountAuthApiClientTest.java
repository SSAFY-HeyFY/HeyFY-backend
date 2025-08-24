package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthCheckResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthResponseDto;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;

@SpringBootTest
class ShinhanAccountAuthApiClientTest {

    @Autowired
    ShinhanAccountAuthApiClient client;

    @TestConfiguration
    static class WebClientTestConfig {
        @Bean
        public WebClient.Builder webClientBuilder() {
            return WebClient.builder();
        }
    }

    private static final String USER_KEY = "e6ab7c69-db18-414e-b9a0-2fedf94eaf7f";
    private static final String ACCOUNT_NO = "0017459003466586";

    @Test
    @DisplayName("계좌 인증 요청 API 실행 테스트")
    void openAccountAuthTest() {
        try {
            AccountAuthResponseDto result = client.openAccountAuth(USER_KEY, ACCOUNT_NO);
            System.out.println("openAccountAuth result: " + result);
        } catch (Exception e) {
            System.out.println("openAccountAuth error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("계좌 인증코드 확인 API 실행 테스트")
    void checkAuthCodeTest() {
        try {
            // 실제 테스트 시 올바른 인증코드로 교체 필요
            String authCode = "4190";
            AccountAuthCheckResponseDto result = client.checkAuthCode(USER_KEY, ACCOUNT_NO, authCode);
            System.out.println("checkAuthCode result: " + result);
        } catch (Exception e) {
            System.out.println("checkAuthCode error: " + e.getMessage());
        }
    }
}

