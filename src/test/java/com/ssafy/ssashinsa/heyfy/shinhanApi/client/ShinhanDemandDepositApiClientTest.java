package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanInquireDemandDepositAccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanUpdateAccountResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireSingleTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferResponseDto;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;

@SpringBootTest
class ShinhanDemandDepositApiClientTest {

    @Autowired
    ShinhanDemandDepositApiClient client;

    @TestConfiguration
    static class WebClientTestConfig {
        @Bean
        public WebClient.Builder webClientBuilder() {
            return WebClient.builder();
        }
    }

//    @Test
//    @DisplayName("예금계좌 생성 API 실행 테스트")
//    void createDemandDepositAccountTest() {
//        try {
//            ShinhanCreateDepositResponseDto result = client.createDemandDepositAccount("e6ab7c69-db18-414e-b9a0-2fedf94eaf7f");
//        } catch (Exception e) {
//            System.out.println("createDemandDepositAccount error: " + e.getMessage());
//        }
//    }

    @Test
    @DisplayName("예금계좌 목록 조회 API 실행 테스트")
    void inquireDemandDepositAccountListTest() {
        try {
            ShinhanInquireDepositResponseDto result = client.inquireDemandDepositAccountList("e6ab7c69-db18-414e-b9a0-2fedf94eaf7f");
        } catch (Exception e) {
            System.out.println("inquireDemandDepositAccountList error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("예금계좌 단건 조회 API 실행 테스트")
    void inquireDemandDepositAccountTest() {
        try {
            ShinhanInquireSingleDepositResponseDto result = client.inquireDemandDepositAccount("e6ab7c69-db18-414e-b9a0-2fedf94eaf7f", "0017459003466586");
        } catch (Exception e) {
            System.out.println("inquireDemandDepositAccount error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("거래내역 목록 조회 API 실행 테스트")
    void inquireTransactionHistoryListTest() {
        try {
            InquireTransactionHistoryResponseDto result = client.inquireTransactionHistoryList("e6ab7c69-db18-414e-b9a0-2fedf94eaf7f", "0017459003466586");
        } catch (Exception e) {
            System.out.println("inquireTransactionHistoryList error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("거래내역 단건 조회 API 실행 테스트")
    void inquireTransactionHistoryTest() {
        try {
            InquireSingleTransactionHistoryResponseDto result = client.inquireTransactionHistory("e6ab7c69-db18-414e-b9a0-2fedf94eaf7f", "0017459003466586", "101686");
        } catch (Exception e) {
            System.out.println("inquireTransactionHistory error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("계좌이체 API 실행 테스트")
    void updateDemandDepositAccountTransferTest() {
        try {
            TransferResponseDto result = client.updateDemandDepositAccountTransfer(
                    "e6ab7c69-db18-414e-b9a0-2fedf94eaf7f", "0017459003466586", "0013177281249735",
                    "10000", "입금메모", "출금메모");
        } catch (Exception e) {
            System.out.println("updateDemandDepositAccountTransfer error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("예금계좌 잔액조회 API 실행 테스트")
    void inquireDemandDepositAccountBalanceTest() {
        try {
            ShinhanInquireDemandDepositAccountBalanceResponseDto result = client.inquireDemandDepositAccountBalance("0017459003466586", "e6ab7c69-db18-414e-b9a0-2fedf94eaf7f");
        } catch (Exception e) {
            System.out.println("inquireDemandDepositAccountBalance error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("예금계좌 입금 API 실행 테스트")
    void updateDemandDepositAccountDepositTest() {
        try {
            ShinhanUpdateAccountResponseDto result = client.updateDemandDepositAccountDeposit("0017459003466586", 10000.0, "e6ab7c69-db18-414e-b9a0-2fedf94eaf7f");
        } catch (Exception e) {
            System.out.println("updateDemandDepositAccountDeposit error: " + e.getMessage());
        }
    }
}
