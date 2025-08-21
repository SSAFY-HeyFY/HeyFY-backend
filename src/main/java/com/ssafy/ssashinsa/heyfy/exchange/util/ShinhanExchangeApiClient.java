package com.ssafy.ssashinsa.heyfy.exchange.util;

import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.AccountBalanceRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanInquireDemandDepositAccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.exception.ExchangeErrorCode;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireDemandDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class ShinhanExchangeApiClient {

    private final ShinhanApiClient apiClient;
    private final ShinhanApiUtil shinhanApiUtil;

    private void logRequest(Object requestDto) {
        try {
            log.info("Request JSON: {}", new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(requestDto));
        } catch (Exception e) {
            log.error("Request logging error", e);
        }
    }
    private void logResponse(Object responseDto) {
        try {
            log.info("Response JSON: {}", new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(responseDto));
        } catch (Exception e) {
            log.error("Response logging error", e);
        }
    }

    public ShinhanInquireDemandDepositResponseDto getAccountInfoFromExternalApi(String accountNo, String userKey) {
        AccountBalanceRequestDto requestDto = createAccountBalanceRequestDto(accountNo, userKey);
        logRequest(requestDto);
        ShinhanInquireDemandDepositResponseDto response = apiClient.getClient("edu")
                .post()
                .uri("/demandDeposit/inquireDemandDepositAccount")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND);
                        }))
                .bodyToMono(ShinhanInquireDemandDepositResponseDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }

    private AccountBalanceRequestDto createAccountBalanceRequestDto(String accountNo, String userKey) {
        String apiKey = apiClient.getManagerKey();
        return AccountBalanceRequestDto.builder()
                .Header(shinhanApiUtil.createHeaderDto("inquireDemandDepositAccountBalance", "inquireDemandDepositAccountBalance", apiKey, userKey))
                .accountNo(accountNo)
                .build();
    }

    private AccountBalanceRequestDto createForeignAccountBalanceRequestDto(String accountNo, String userKey) {
        String apiKey = apiClient.getManagerKey();
        return AccountBalanceRequestDto.builder()
                .Header(shinhanApiUtil.createHeaderDto("inquireForeignCurrencyDemandDepositAccountBalance", "inquireForeignCurrencyDemandDepositAccountBalance", apiKey, userKey))
                .accountNo(accountNo)
                .build();
    }


    public ShinhanInquireDemandDepositAccountBalanceResponseDto getAccountBalanceFromExternalApi(String accountNo, String userKey) {
        AccountBalanceRequestDto requestDto = createAccountBalanceRequestDto(accountNo, userKey);
        logRequest(requestDto);
        ShinhanInquireDemandDepositAccountBalanceResponseDto response = apiClient.getClient("edu")
                .post()
                .uri("/demandDeposit/inquireDemandDepositAccountBalance")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND);
                        }))
                .bodyToMono(ShinhanInquireDemandDepositAccountBalanceResponseDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }

    public ShinhanInquireDemandDepositAccountBalanceResponseDto getForeignAccountBalanceFromExternalApi(String accountNo, String userKey) {
        AccountBalanceRequestDto requestDto = createForeignAccountBalanceRequestDto(accountNo, userKey);
        logRequest(requestDto);
        ShinhanInquireDemandDepositAccountBalanceResponseDto response = apiClient.getClient("edu")
                .post()
                .uri("/demandDeposit/inquireForeignCurrencyDemandDepositAccountBalance")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND);
                        }))
                .bodyToMono(ShinhanInquireDemandDepositAccountBalanceResponseDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }
}
