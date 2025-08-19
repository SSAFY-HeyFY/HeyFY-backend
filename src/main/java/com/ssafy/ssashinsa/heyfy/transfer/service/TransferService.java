package com.ssafy.ssashinsa.heyfy.transfer.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.transfer.dto.*;
import com.ssafy.ssashinsa.heyfy.transfer.exception.TransferApiErrorCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClientResponseException;

@Service
@Slf4j
@RequiredArgsConstructor
public class TransferService {

    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ObjectMapper objectMapper = new ObjectMapper();

    private static final String TEMP_USER_KEY = "37c844c5-9b24-4daa-becb-ca52763a7b39";

    public EntireTransferResponseDto callTransfer(String withdrawalAccountNo, String depositAccountNo, long amount) {
        TransferRequestDto requestDto = createRequestDto(withdrawalAccountNo, depositAccountNo, amount);
        logRequest(requestDto);

        try {
            return shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/updateDemandDepositAccountTransfer")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .bodyToMono(EntireTransferResponseDto.class)
                    .block();

        } catch (WebClientResponseException e) {
            String errorBody = e.getResponseBodyAsString();
            log.error("API returned an error status. Body: {}", errorBody);

            ShinhanApiErrorResponseDto errorDto;
            try {
                errorDto = objectMapper.readValue(errorBody, ShinhanApiErrorResponseDto.class);
            } catch (Exception parseException) {
                log.error("Failed to parse the error response body.", parseException);
                throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
            }

            String responseCode = errorDto.getResponseCode();
            TransferApiErrorCode errorCode = TransferApiErrorCode.fromCode(responseCode);
            throw new CustomException(errorCode);
        }
    }

    private TransferRequestDto createRequestDto(String withdrawalAccountNo, String depositAccountNo, long amount) {
        String apiKey = shinhanApiClient.getManagerKey();
        ShinhanCommonRequestHeaderDto headerDto = shinhanApiUtil.createHeaderDto(
                "updateDemandDepositAccountTransfer", "updateDemandDepositAccountTransfer", apiKey, TEMP_USER_KEY);
        return TransferRequestDto.builder()
                .withdrawalAccountNo(withdrawalAccountNo)
                .depositAccountNo(depositAccountNo)
                .transactionBalance(String.valueOf(amount))
                .Header(headerDto)
                .build();
    }

    private void logRequest(Object requestDto) {
        try {
            log.info("Request JSON: {}", new ObjectMapper().writeValueAsString(requestDto));
        } catch (Exception e) {
            log.error("Request logging error", e);
        }
    }

    private void logResponse(Object responseDto) {
        try {
            log.info("Response JSON: {}", new ObjectMapper().writeValueAsString(responseDto));
        } catch (Exception e) {
            log.error("Response logging error", e);
        }
    }
}