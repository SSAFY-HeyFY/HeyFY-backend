package com.ssafy.ssashinsa.heyfy.transfer.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.account.exception.AccountErrorCode;
import com.ssafy.ssashinsa.heyfy.account.service.AccountService;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanErrorResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.transfer.dto.EntireTransferResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferRequestDto;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.ClientResponse;
import reactor.core.publisher.Mono;

@Service
@Slf4j
@RequiredArgsConstructor
public class TransferService {

    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final UserRepository userRepository;
    private final AccountService accountService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public EntireTransferResponseDto callTransfer(String depositAccountNo, String amount, String transactionSummary) {
        Users user = findCurrentUser();

        String withdrawalAccountNo = accountService.getAccounts()
                .orElseThrow(() -> new CustomException(AccountErrorCode.WITHDRAWAL_ACCOUNT_NOT_FOUND))
                .getAccount()
                .getAccountNo();

        String serviceCode = "updateDemandDepositAccountTransfer";
        TransferRequestDto requestDto = createRequestDto(withdrawalAccountNo, depositAccountNo, amount, user.getUserKey(), serviceCode, transactionSummary);

        String uri = "/demandDeposit/updateDemandDepositAccountTransfer";
        return executeTransferApi(uri, requestDto);
    }

    public EntireTransferResponseDto callForeignTransfer(String depositAccountNo, String amount, String transactionSummary) {
        Users user = findCurrentUser();

        String withdrawalAccountNo = accountService.getAccounts()
                .orElseThrow(() -> new CustomException(AccountErrorCode.WITHDRAWAL_ACCOUNT_NOT_FOUND))
                .getForeignAccount()
                .getAccountNo();

        String serviceCode = "updateForeignCurrencyDemandDepositAccountTransfer";
        TransferRequestDto requestDto = createRequestDto(withdrawalAccountNo, depositAccountNo, amount, user.getUserKey(), serviceCode, transactionSummary);

        String uri = "/demandDeposit/foreignCurrency/updateForeignCurrencyDemandDepositAccountTransfer";
        return executeTransferApi(uri, requestDto);
    }

    private EntireTransferResponseDto executeTransferApi(String uri, TransferRequestDto requestDto) {
        try {
            logRequest(requestDto);
            return shinhanApiClient.getClient("edu")
                    .post()
                    .uri(uri)
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, this::handleApiError) // 에러 핸들링 로직 분리
                    .bodyToMono(EntireTransferResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();
        } catch (CustomException e) {
            throw e;
        } catch (Exception e) {
            log.error("Transfer API 호출 중 알 수 없는 오류 발생: {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    private Mono<? extends Throwable> handleApiError(ClientResponse response) {
        // DTO 파싱 성공 시의 에러 처리 로직
        Mono<Throwable> successCase = response.bodyToMono(ShinhanErrorResponseDto.class)
                .map(errorBody -> {
                    String errorMessage = String.format("[%s] %s", errorBody.getResponseCode(), errorBody.getResponseMessage());
                    ErrorCode errorCode;
                    if (response.statusCode().is4xxClientError()) {
                        errorCode = ShinhanApiErrorCode.API_INVALID_REQUEST;
                    } else {
                        errorCode = ShinhanApiErrorCode.API_CALL_FAILED;
                    }
                    return new CustomException(errorCode, errorMessage);
                });

        // DTO 파싱 실패 시의 에러 처리 로직 (body를 String으로 읽어 로그 남김)
        Mono<Throwable> failureCase = response.bodyToMono(String.class)
                .map(rawBody -> {
                    log.error("Failed to parse Shinhan API error response. Raw body: {}", rawBody);
                    return new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
                });

        // successCase를 시도하고, 실패하면(onErrorResume) failureCase를 실행
        return successCase.onErrorResume(e -> failureCase);
    }

    private Users findCurrentUser() {
        String studentId = SecurityUtil.getCurrentStudentId();
        if (studentId == null || studentId.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND);
        }
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));
        if (user.getUserKey() == null || user.getUserKey().isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }
        return user;
    }


    private TransferRequestDto createRequestDto(String withdrawalAccountNo, String depositAccountNo, String amount, String userKey, String serviceCode, String transactionSummary) {
        String apiKey = shinhanApiClient.getManagerKey();
        ShinhanCommonRequestHeaderDto headerDto = shinhanApiUtil.createHeaderDto(serviceCode, serviceCode, apiKey, userKey);
        return TransferRequestDto.builder()
                .withdrawalAccountNo(withdrawalAccountNo)
                .depositAccountNo(depositAccountNo)
                .transactionBalance(String.valueOf(amount))
                .depositTransactionSummary(transactionSummary)
                .withdrawalTransactionSummary(transactionSummary)
                .Header(headerDto)
                .build();
    }

    private void logRequest(Object requestDto) {
        try {
            log.info("Request JSON: {}", objectMapper.writeValueAsString(requestDto));
        } catch (JsonProcessingException e) {
            log.error("Request logging error", e);
        }
    }

    private void logResponse(Object responseDto) {
        try {
            log.info("Response JSON: {}", objectMapper.writeValueAsString(responseDto));
        } catch (JsonProcessingException e) {
            log.error("Response logging error", e);
        }
    }
}
