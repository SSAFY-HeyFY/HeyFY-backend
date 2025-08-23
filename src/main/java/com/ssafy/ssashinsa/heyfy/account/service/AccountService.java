package com.ssafy.ssashinsa.heyfy.account.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.account.dto.*;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Optional;

@Service
@Slf4j
@RequiredArgsConstructor
public class AccountService {

    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository foreignAccountRepository;
    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public Optional<AccountPairDto> getAccounts() {
        String studentId = SecurityUtil.getCurrentStudentId();
//        String userEmail = userRepository.findByStudentId(studentId)
//                .orElseThrow(() -> new CustomException(AuthErrorCode.USER_NOT_FOUND))
//                .getEmail();
        String userEmail = SecurityUtil.getCurrentUserEmail();
        return userRepository.findAccountsByUserEmail(userEmail);
    }

    public AccountAuthResponseDto AccountAuth() {
        try {
            String apiKey = shinhanApiClient.getManagerKey();
            String accountTypeUniqueNo = shinhanApiClient.getForeignAccountTypeUniqueNo();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "openAccountAuth",
                    "openAccountAuth",
                    apiKey,
                    userKey
            );

            String accountNo = accountRepository.findByUser(user)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND))
                    .getAccountNo();

            AccountAuthRequestDto requestDto = AccountAuthRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountNo(accountNo)
                    .authText("SSAFY")
                    .build();

            logRequest(requestDto);

            AccountAuthResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/accountAuth/openAccountAuth")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(AccountAuthResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public AccountAuthResponseDto AccountAuth(String accountNo) {
        try {
            String apiKey = shinhanApiClient.getManagerKey();
            String accountTypeUniqueNo = shinhanApiClient.getForeignAccountTypeUniqueNo();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "openAccountAuth",
                    "openAccountAuth",
                    apiKey,
                    userKey
            );

            AccountAuthRequestDto requestDto = AccountAuthRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountNo(accountNo)
                    .authText("SSAFY")
                    .build();

            logRequest(requestDto);

            AccountAuthResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/accountAuth/openAccountAuth")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(AccountAuthResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public AccountAuthCheckResponseDto accountAuthCheck(String accountNo, String authCode) {
        try {
            String apiKey = shinhanApiClient.getManagerKey();
            String accountTypeUniqueNo = shinhanApiClient.getForeignAccountTypeUniqueNo();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "checkAuthCode",
                    "checkAuthCode",
                    apiKey,
                    userKey
            );

            AccountAuthCheckRequestDto requestDto = AccountAuthCheckRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountNo(accountNo)
                    .authText("SSAFY")
                    .authCode(authCode)
                    .build();

            logRequest(requestDto);

            AccountAuthCheckResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/accountAuth/checkAuthCode")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(AccountAuthCheckResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public InquireTransactionHistoryResponseDto getTransactionHistory() {
        try {
            String apiKey = shinhanApiClient.getManagerKey();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "inquireTransactionHistoryList",
                    "inquireTransactionHistoryList",
                    apiKey,
                    userKey
            );

            String accountNo = accountRepository.findByUser(user)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND))
                    .getAccountNo();

            LocalDateTime now = LocalDateTime.now();
            InquireTransactionHistoryRequestDto requestDto = InquireTransactionHistoryRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountNo(accountNo)
                    .startDate("20230101")
                    .endDate(now.format(DateTimeFormatter.ofPattern("yyyyMMdd")))
                    .transactionType("A")
                    .orderByType("DESC")
                    .build();

            logRequest(requestDto);

            InquireTransactionHistoryResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/inquireTransactionHistoryList")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(InquireTransactionHistoryResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            log.error("계좌 개설 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public InquireTransactionHistoryResponseDto getTransactionHistory(String accountNo) {
        try {
            String apiKey = shinhanApiClient.getManagerKey();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "inquireTransactionHistoryList",
                    "inquireTransactionHistoryList",
                    apiKey,
                    userKey
            );

            Account account = accountRepository.findByUserAndAccountNo(user, accountNo)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND));

            LocalDateTime now = LocalDateTime.now();
            InquireTransactionHistoryRequestDto requestDto = InquireTransactionHistoryRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountNo(accountNo)
                    .startDate("20230101")
                    .endDate(now.format(DateTimeFormatter.ofPattern("yyyyMMdd")))
                    .transactionType("A")
                    .orderByType("DESC")
                    .build();

            logRequest(requestDto);

            InquireTransactionHistoryResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/inquireTransactionHistoryList")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(InquireTransactionHistoryResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            log.error("계좌 개설 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public InquireSingleTransactionHistoryResponseDto getSingleTransactionHistory(String accountNo, String transactionUniqueNo) {
        try {
            String apiKey = shinhanApiClient.getManagerKey();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "inquireTransactionHistory",
                    "inquireTransactionHistory",
                    apiKey,
                    userKey
            );

            InquireSingleTransactionHistoryRequestDto requestDto = InquireSingleTransactionHistoryRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountNo(accountNo)
                    .transactionUniqueNo(transactionUniqueNo)
                    .build();

            logRequest(requestDto);

            InquireSingleTransactionHistoryResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/inquireTransactionHistory")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(InquireSingleTransactionHistoryResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            log.error("계좌 개설 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    public InquireTransactionHistoryResponseDto getForeignTransactionHistory(String accountNo) {
        try {
            String apiKey = shinhanApiClient.getManagerKey();

            String studentId = SecurityUtil.getCurrentStudentId();
            Users user = userRepository.findByStudentId(studentId)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "inquireForeignCurrencyTransactionHistoryList",
                    "inquireForeignCurrencyTransactionHistoryList",
                    apiKey,
                    userKey
            );

            ForeignAccount account = foreignAccountRepository.findByUserAndAccountNo(user, accountNo)
                    .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_NOT_FOUND));

            LocalDateTime now = LocalDateTime.now();
            InquireTransactionHistoryRequestDto requestDto = InquireTransactionHistoryRequestDto.builder()
                    .Header(commonHeaderDto)
                    .accountNo(accountNo)
                    .startDate("20230101")
                    .endDate(now.format(DateTimeFormatter.ofPattern("yyyyMMdd")))
                    .transactionType("A")
                    .orderByType("DESC")
                    .build();

            logRequest(requestDto);

            InquireTransactionHistoryResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/foreignCurrency/inquireForeignCurrencyTransactionHistoryList")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanRegisterApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(InquireTransactionHistoryResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

        } catch (CustomException ce) {
            log.error("커스텀 예외 발생: {}", ce.getMessage());
            throw ce;
        } catch (Exception e) {
            log.error("계좌 개설 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }



    private void logRequest(Object requestDto) {
        try {
            log.info("Request JSON: {}", objectMapper.writeValueAsString(requestDto));
        } catch (Exception e) {
            log.error("Request logging error", e);
        }
    }

    private void logResponse(Object responseDto) {
        try {
            log.info("Response JSON: {}", objectMapper.writeValueAsString(responseDto));
        } catch (Exception e) {
            log.error("Response logging error", e);
        }
    }

}