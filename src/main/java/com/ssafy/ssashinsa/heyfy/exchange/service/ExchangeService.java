package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.AIPredictionResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.AccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.ExchangePageResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchange.HistoricalAnalysisResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.AccountBalanceRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanInquireDemandDepositAccountBalanceResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.exception.ExchangeErrorCode;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireDemandDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;


@Slf4j
@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class ExchangeService {

    private final ShinhanApiClient apiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository foreignAccountRepository;

    @Transactional
    public AccountBalanceResponseDto getAccountBalance(String studentId) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        Account account = accountRepository.findAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));


        ShinhanInquireDemandDepositAccountBalanceResponseDto response = getAccountBalanceFromExternalApi(account.getAccountNo(), user.getUserKey());

        return AccountBalanceResponseDto.builder()
                .accountNo(account.getAccountNo())
                .accountBalance(Integer.parseInt(response.getREC().getAccountBalance()))
                .currency("KRW")
                .build();
    }

    @Transactional
    public HistoricalAnalysisResponseDto getHIstoriaclAnalysis(){
        return HistoricalAnalysisResponseDto.builder()
                .message("Over the past 30 days, today shows the highest exchange rate")
                .build();
    }

    @Transactional
    public AIPredictionResponseDto getExchangeRateAIPrediction(){
        return AIPredictionResponseDto.builder()
                .message("The rate may increase by $0.54 more in the near future")
                .build();
    }

    @Transactional
    public ExchangePageResponseDto getExchangePage(String studentId) {
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND));
        Account account = accountRepository.findAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));
        ForeignAccount foreignAccount = foreignAccountRepository.findForeignAccountByUserEmail(user.getEmail())
                .orElseThrow(() -> new CustomException(ExchangeErrorCode.ACCOUNT_NOT_FOUND));

        // account balance
        ShinhanInquireDemandDepositAccountBalanceResponseDto accountBalance
                = getAccountBalanceFromExternalApi(account.getAccountNo(), user.getUserKey());
        AccountBalanceResponseDto accountBalanceResponseDto = AccountBalanceResponseDto.builder()
                .accountNo(accountBalance.getREC().getAccountNo())
                .accountBalance(Integer.parseInt(accountBalance.getREC().getAccountBalance()))
                .currency(accountBalance.getREC().getCurrency())
                .isForeign(false)
                .build();

        // foreign account balance
        ShinhanInquireDemandDepositAccountBalanceResponseDto foreignAccountBalance
                = getAccountBalanceFromExternalApi(foreignAccount.getAccountNo(), user.getUserKey());
        AccountBalanceResponseDto foreignAccountBalanceResponseDto = AccountBalanceResponseDto.builder()
                .accountNo(foreignAccountBalance.getREC().getAccountNo())
                .accountBalance(Integer.parseInt(foreignAccountBalance.getREC().getAccountBalance()))
                .currency(foreignAccountBalance.getREC().getCurrency())
                .isForeign(true)
                .build();

        // AI Prediction
        AIPredictionResponseDto aiPredictionResponseDto = AIPredictionResponseDto.builder()
                .message("The rate may increase by $0.54 more in the near future")
                .build();
        // Historical Analysis
        HistoricalAnalysisResponseDto historicalAnalysisResponseDto = HistoricalAnalysisResponseDto.builder()
                .message("Over the past 30 days, today shows the highest exchange rate")
                .build();
        return ExchangePageResponseDto.builder()
                .aiPrediction(aiPredictionResponseDto)
                .historicalAnalysis(historicalAnalysisResponseDto)
                .accountBalance(accountBalanceResponseDto)
                .foreignAccountBalance(foreignAccountBalanceResponseDto)
                .build();
    }

    private ShinhanInquireDemandDepositAccountBalanceResponseDto getAccountBalanceFromExternalApi(String accountNo, String userKey) {
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
                            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
                        }))
                .bodyToMono(ShinhanInquireDemandDepositAccountBalanceResponseDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }

    private ShinhanInquireDemandDepositResponseDto getAccountInfoFromExternalApi(String accountNo, String userKey) {
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
                            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
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
}
