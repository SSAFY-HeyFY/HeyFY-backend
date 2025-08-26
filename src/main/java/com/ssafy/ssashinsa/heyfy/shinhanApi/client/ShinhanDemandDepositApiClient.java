package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.AccountBalanceRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanInquireDemandDepositAccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanUpdateAccountRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanUpdateAccountResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireSingleTransactionHistoryRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireSingleTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanException;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Component
@RequiredArgsConstructor
public class ShinhanDemandDepositApiClient {
    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;

    public ShinhanCreateDepositResponseDto createDemandDepositAccount(String userKey) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("createDemandDepositAccount", "createDemandDepositAccount", userKey);
        ShinhanCreateDepositRequestDto requestDto = ShinhanCreateDepositRequestDto.builder()
                .Header(header)
                .accountTypeUniqueNo(shinhanApiClient.getAccountTypeUniqueNo())
                .build();
        shinhanApiUtil.logRequest(requestDto);
        ShinhanCreateDepositResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/createDemandDepositAccount")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(ShinhanCreateDepositResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public ShinhanInquireDepositResponseDto inquireDemandDepositAccountList(String userKey) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireDemandDepositAccountList", "inquireDemandDepositAccountList", userKey);
        ShinhanInquireDepositRequestDto requestDto = ShinhanInquireDepositRequestDto.builder()
                .Header(header)
                .build();
        shinhanApiUtil.logRequest(requestDto);

        ShinhanInquireDepositResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/inquireDemandDepositAccountList")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(ShinhanInquireDepositResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public ShinhanInquireSingleDepositResponseDto inquireDemandDepositAccount(String userKey, String accountNo) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireDemandDepositAccount", "inquireDemandDepositAccount", userKey);
        ShinhanInquireSingleDepositRequestDto requestDto = ShinhanInquireSingleDepositRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .build();
        shinhanApiUtil.logRequest(requestDto);

        ShinhanInquireSingleDepositResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/inquireDemandDepositAccount")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(ShinhanInquireSingleDepositResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public ShinhanInquireSingleDepositResponseDto inquireDemandForeignDepositAccount(String userKey, String accountNo) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireForeignCurrencyDemandDepositAccount", "inquireForeignCurrencyDemandDepositAccount", userKey);
        ShinhanInquireSingleDepositRequestDto requestDto = ShinhanInquireSingleDepositRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .build();
        shinhanApiUtil.logRequest(requestDto);

        ShinhanInquireSingleDepositResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/foreignCurrency/inquireForeignCurrencyDemandDepositAccount")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(ShinhanInquireSingleDepositResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public InquireTransactionHistoryResponseDto inquireTransactionHistoryList(String userKey, String accountNo) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireTransactionHistoryList", "inquireTransactionHistoryList", userKey);
        LocalDateTime now = LocalDateTime.now();
        InquireTransactionHistoryRequestDto requestDto = InquireTransactionHistoryRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .startDate("20230101")
                .endDate(now.format(DateTimeFormatter.ofPattern("yyyyMMdd")))
                .transactionType("A")
                .orderByType("DESC")
                .build();
        shinhanApiUtil.logRequest(requestDto);

        InquireTransactionHistoryResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/inquireTransactionHistoryList")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(InquireTransactionHistoryResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public InquireTransactionHistoryResponseDto inquireForeignTransactionHistoryList(String userKey, String accountNo) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireForeignCurrencyTransactionHistoryList", "inquireForeignCurrencyTransactionHistoryList", userKey);
        LocalDateTime now = LocalDateTime.now();
        InquireTransactionHistoryRequestDto requestDto = InquireTransactionHistoryRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .startDate("20230101")
                .endDate(now.format(DateTimeFormatter.ofPattern("yyyyMMdd")))
                .transactionType("A")
                .orderByType("DESC")
                .build();
        shinhanApiUtil.logRequest(requestDto);

        InquireTransactionHistoryResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/foreignCurrency/inquireForeignCurrencyTransactionHistoryList")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(InquireTransactionHistoryResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public InquireSingleTransactionHistoryResponseDto inquireTransactionHistory(String userKey, String accountNo, String transactionUniqueNo) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireTransactionHistory", "inquireTransactionHistory", userKey);
        InquireSingleTransactionHistoryRequestDto requestDto = InquireSingleTransactionHistoryRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .transactionUniqueNo(transactionUniqueNo)
                .build();
        shinhanApiUtil.logRequest(requestDto);

        InquireSingleTransactionHistoryResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/inquireTransactionHistory")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(InquireSingleTransactionHistoryResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public TransferResponseDto updateDemandDepositAccountTransfer(
            String userKey, String withdrawalAccountNo, String depositAccountNo,
            String transactionBalance, String depositTransactionSummary,
            String withdrawalTransactionSummary) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("updateDemandDepositAccountTransfer", "updateDemandDepositAccountTransfer", userKey);

        TransferRequestDto requestDto = TransferRequestDto.builder()
                .Header(header)
                .withdrawalAccountNo(withdrawalAccountNo)
                .depositAccountNo(depositAccountNo)
                .transactionBalance(transactionBalance)
                .depositTransactionSummary(depositTransactionSummary)
                .withdrawalTransactionSummary(withdrawalTransactionSummary)
                .build();
        shinhanApiUtil.logRequest(requestDto);

        TransferResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/updateDemandDepositAccountTransfer")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(TransferResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public ShinhanInquireDemandDepositAccountBalanceResponseDto inquireDemandDepositAccountBalance(String accountNo, String userKey) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireDemandDepositAccountBalance", "inquireDemandDepositAccountBalance", userKey);
        AccountBalanceRequestDto requestDto = AccountBalanceRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .build();

        shinhanApiUtil.logRequest(requestDto);
        ShinhanInquireDemandDepositAccountBalanceResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/inquireDemandDepositAccountBalance")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(ShinhanInquireDemandDepositAccountBalanceResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }

    public ShinhanUpdateAccountResponseDto updateDemandDepositAccountDeposit(String accountNo, Double transactionBalance, String userKey) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("updateDemandDepositAccountDeposit", "updateDemandDepositAccountDeposit", userKey);
        ShinhanUpdateAccountRequestDto requestDto = ShinhanUpdateAccountRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .transactionBalance(transactionBalance)
                .build();

        shinhanApiUtil.logRequest(requestDto);
        ShinhanUpdateAccountResponseDto response = shinhanApiClient.getClient("demand-deposit")
                .post()
                .uri("/updateDemandDepositAccountDeposit")
                .header("Content-Type", "application/json")
                .bodyValue(requestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            String responseCode = shinhanApiUtil.getResponseCode(body);
                            throw new ShinhanException(ShinhanErrorCode.valueOf(responseCode));
                        }))
                .bodyToMono(ShinhanUpdateAccountResponseDto.class)
                .doOnNext(shinhanApiUtil::logResponse)
                .block();
        return response;
    }
}

