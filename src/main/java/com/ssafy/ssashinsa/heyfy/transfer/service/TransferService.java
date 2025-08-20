package com.ssafy.ssashinsa.heyfy.transfer.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.account.exception.AccountErrorCode;
import com.ssafy.ssashinsa.heyfy.account.service.AccountService;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanErrorResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.transfer.dto.EntireTransferResponseDto;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferRequestDto;
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

    public EntireTransferResponseDto callTransfer(String depositAccountNo, long amount) {
        Users user = findCurrentUser();

        String withdrawalAccountNo = accountService.getAccounts()
                .orElseThrow(() -> new CustomException(AccountErrorCode.WITHDRAWAL_ACCOUNT_NOT_FOUND))
                .getAccount()
                .getAccountNo();

        String serviceCode = "updateDemandDepositAccountTransfer";
        TransferRequestDto requestDto = createRequestDto(withdrawalAccountNo, depositAccountNo, amount, user.getUserKey(), serviceCode);

        String uri = "/demandDeposit/updateDemandDepositAccountTransfer";
        return executeTransferApi(uri, requestDto);
    }

    public EntireTransferResponseDto callForeignTransfer(String depositAccountNo, long amount) {
        Users user = findCurrentUser();

        String withdrawalAccountNo = accountService.getAccounts()
                .orElseThrow(() -> new CustomException(AccountErrorCode.WITHDRAWAL_ACCOUNT_NOT_FOUND))
                .getForeignAccount()
                .getAccountNo();

        String serviceCode = "updateForeignCurrencyDemandDepositAccountTransfer";
        TransferRequestDto requestDto = createRequestDto(withdrawalAccountNo, depositAccountNo, amount, user.getUserKey(), serviceCode);

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
        return response.bodyToMono(ShinhanErrorResponseDto.class)
                .map(errorBody -> {
                    log.error("External API Error Body: {}", errorBody);
                    String errorMessage = String.format("[%s] %s", errorBody.getResponseCode(), errorBody.getResponseMessage());
                    return new CustomException(ShinhanApiErrorCode.API_CALL_FAILED, errorMessage);
                })
                .onErrorResume(e ->
                        response.bodyToMono(String.class)
                                .map(body -> {
                                    log.error("Failed to parse error response. Raw body: {}", body, e);
                                    return new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
                                })
                )
                .flatMap(Mono::error);
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


    private TransferRequestDto createRequestDto(String withdrawalAccountNo, String depositAccountNo, long amount, String userKey, String serviceCode) {
        String apiKey = shinhanApiClient.getManagerKey();
        ShinhanCommonRequestHeaderDto headerDto = shinhanApiUtil.createHeaderDto(serviceCode, serviceCode, apiKey, userKey);
        return TransferRequestDto.builder()
                .withdrawalAccountNo(withdrawalAccountNo)
                .depositAccountNo(depositAccountNo)
                .transactionBalance(String.valueOf(amount))
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