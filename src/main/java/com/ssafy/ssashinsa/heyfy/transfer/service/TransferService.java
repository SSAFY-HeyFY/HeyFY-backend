package com.ssafy.ssashinsa.heyfy.transfer.service;

import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import com.ssafy.ssashinsa.heyfy.transfer.dto.EntireTransferResponseDto;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferRequestDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Service
@Slf4j
@RequiredArgsConstructor
public class TransferService {

    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;

    // TODO: 추후 DB에서 사용자 정보를 조회하여 userKey를 가져오도록 수정
    private static final String TEMP_USER_KEY = "37c844c5-9b24-4daa-becb-ca52763a7b39";

    public EntireTransferResponseDto callTransfer(String withdrawalAccountNo, String depositAccountNo, long amount) {
        try {
            TransferRequestDto requestDto = createRequestDto(withdrawalAccountNo, depositAccountNo, amount);
            logRequest(requestDto);

            EntireTransferResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/updateDemandDepositAccountTransfer")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(EntireTransferResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            logResponse(response);
            return response;
        } catch (Exception e) {
            log.error("계좌 이체 API 호출 실패 : {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }


    private TransferRequestDto createRequestDto(String withdrawalAccountNo, String depositAccountNo, long amount) {
        String apiKey = shinhanApiClient.getManagerKey();

        ShinhanCommonRequestHeaderDto headerDto = shinhanApiUtil.createHeaderDto(
                "updateDemandDepositAccountTransfer",
                "updateDemandDepositAccountTransfer",
                apiKey,
                TEMP_USER_KEY
        );

        return TransferRequestDto.builder()
                .withdrawalAccountNo(withdrawalAccountNo)
                .depositAccountNo(depositAccountNo)
                .transactionBalance(String.valueOf(amount))
                .Header(headerDto)
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