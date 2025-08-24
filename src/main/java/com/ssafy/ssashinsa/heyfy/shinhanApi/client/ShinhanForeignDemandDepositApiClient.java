package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.AccountBalanceRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanInquireDemandDepositAccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanUpdateAccountRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanUpdateAccountResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.foreign.ShinhanCreateforeignDepositRequestDto;
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
public class ShinhanForeignDemandDepositApiClient {
    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;

    public ShinhanCreateDepositResponseDto createForeignCurrencyDemandDepositAccount(String userKey) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("createForeignCurrencyDemandDepositAccount", "createForeignCurrencyDemandDepositAccount", userKey);

        ShinhanCreateforeignDepositRequestDto requestDto = ShinhanCreateforeignDepositRequestDto.builder()
                .Header(header)
                .accountTypeUniqueNo(shinhanApiClient.getForeignAccountTypeUniqueNo())
                .currency("USD") // 예시로 USD를 사용, 필요에 따라 변경
                .build();

        shinhanApiUtil.logRequest(requestDto);

        ShinhanCreateDepositResponseDto response = shinhanApiClient.getClient("foreign")
                .post()
                .uri("/createForeignCurrencyDemandDepositAccount")
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

    public ShinhanInquireSingleDepositResponseDto inquireForeignCurrencyDemandDepositAccount(String userKey, String accountNo) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireForeignCurrencyDemandDepositAccount", "inquireForeignCurrencyDemandDepositAccount", userKey);

        ShinhanInquireSingleDepositRequestDto requestDto = ShinhanInquireSingleDepositRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .build();

        shinhanApiUtil.logRequest(requestDto);

        ShinhanInquireSingleDepositResponseDto response = shinhanApiClient.getClient("foreign")
                .post()
                .uri("/inquireForeignCurrencyDemandDepositAccount")
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

    public InquireTransactionHistoryResponseDto inquireForeignCurrencyTransactionHistoryList(String userKey, String accountNo) {
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

        InquireTransactionHistoryResponseDto response = shinhanApiClient.getClient("foreign")
                .post()
                .uri("/inquireForeignCurrencyTransactionHistoryList")
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

    public TransferResponseDto updateForeignCurrencyDemandDepositAccountTransfer(
            String userKey, String withdrawalAccountNo, String depositAccountNo,
            String transactionBalance, String depositTransactionSummary,
            String withdrawalTransactionSummary) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("updateForeignCurrencyDemandDepositAccountTransfer", "updateForeignCurrencyDemandDepositAccountTransfer", userKey);

        TransferRequestDto requestDto = TransferRequestDto.builder()
                .Header(header)
                .withdrawalAccountNo(withdrawalAccountNo)
                .depositAccountNo(depositAccountNo)
                .transactionBalance(transactionBalance)
                .depositTransactionSummary(depositTransactionSummary)
                .withdrawalTransactionSummary(withdrawalTransactionSummary)
                .build();
        shinhanApiUtil.logRequest(requestDto);

        TransferResponseDto response = shinhanApiClient.getClient("foreign")
                .post()
                .uri("/updateForeignCurrencyDemandDepositAccountTransfer")
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

    public ShinhanInquireDemandDepositAccountBalanceResponseDto inquireForeignCurrencyDemandDepositAccountBalance(String accountNo, String userKey) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("inquireForeignCurrencyDemandDepositAccountBalance", "inquireForeignCurrencyDemandDepositAccountBalance", userKey);
        AccountBalanceRequestDto requestDto = AccountBalanceRequestDto.builder()
                .accountNo(accountNo)
                .build();
        shinhanApiUtil.logRequest(requestDto);
        ShinhanInquireDemandDepositAccountBalanceResponseDto response = shinhanApiClient.getClient("foreign")
                .post()
                .uri("/inquireForeignCurrencyDemandDepositAccountBalance")
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

    public ShinhanUpdateAccountResponseDto updateForeignCurrencyDemandDepositAccountDeposit(String accountNo, Double transactionBalance, String userKey) {
        ShinhanCommonRequestHeaderDto header = shinhanApiUtil.createHeaderDto("updateForeignCurrencyDemandDepositAccountDeposit", "updateForeignCurrencyDemandDepositAccountDeposit", userKey);
        ShinhanUpdateAccountRequestDto requestDto = ShinhanUpdateAccountRequestDto.builder()
                .Header(header)
                .accountNo(accountNo)
                .transactionBalance(transactionBalance)
                .build();

        shinhanApiUtil.logRequest(requestDto);

        ShinhanUpdateAccountResponseDto response = shinhanApiClient.getClient("foreign")
                .post()
                .uri("/updateForeignCurrencyDemandDepositAccountDeposit")
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
