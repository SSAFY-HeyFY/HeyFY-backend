package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;

@SpringBootTest
class ShinhanForeignDemandDepositApiClientTest {

    @Autowired
    ShinhanForeignDemandDepositApiClient client;

    @TestConfiguration
    static class WebClientTestConfig {
        @Bean
        public WebClient.Builder webClientBuilder() {
            return WebClient.builder();
        }
    }

    private static final String USER_KEY = "e6ab7c69-db18-414e-b9a0-2fedf94eaf7f";
    private static final String ACCOUNT_NO = "0014277609392958";

    @Test
    @DisplayName("외화 예금계좌 생성 API 실행 테스트")
    void createForeignCurrencyDemandDepositAccountTest() {
        try {
            ShinhanCreateDepositResponseDto result = client.createForeignCurrencyDemandDepositAccount(USER_KEY);
        } catch (Exception e) {
            System.out.println("createForeignCurrencyDemandDepositAccount error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("외화 예금계좌 단건 조회 API 실행 테스트")
    void inquireForeignCurrencyDemandDepositAccountTest() {
        try {
            ShinhanInquireSingleDepositResponseDto result = client.inquireForeignCurrencyDemandDepositAccount(USER_KEY, ACCOUNT_NO);
        } catch (Exception e) {
            System.out.println("inquireForeignCurrencyDemandDepositAccount error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("외화 거래내역 목록 조회 API 실행 테스트")
    void inquireForeignCurrencyTransactionHistoryListTest() {
        try {
            InquireTransactionHistoryResponseDto result = client.inquireForeignCurrencyTransactionHistoryList(USER_KEY, ACCOUNT_NO);
        } catch (Exception e) {
            System.out.println("inquireForeignCurrencyTransactionHistoryList error: " + e.getMessage());
        }
    }
}

