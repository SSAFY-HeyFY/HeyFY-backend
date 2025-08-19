package com.ssafy.ssashinsa.heyfy.register.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.register.dto.ShinhanCreateDepositRequestDto;
import com.ssafy.ssashinsa.heyfy.register.dto.ShinhanCreateDepositRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.register.dto.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Service
@Slf4j
@RequiredArgsConstructor
public class RegisterService {

    @Value("${shinhan.manager-key}")
    private String managerKey;

    @Value("${shinhan.temp-user-key}")
    private String TEMP_USER_KEY;


    private final ShinhanApiClient shinhanApiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ObjectMapper objectMapper = new ObjectMapper();


    private static final String ACCOUNT_TYPE_UNIQUE_NO = "001-1-5ca485c2547242";

    public ShinhanCreateDepositResponseDto createDepositAccount() {
        try {
            String apiKey = managerKey;
            System.out.println("매니저 키: "    + apiKey);

            ShinhanCommonRequestHeaderDto commonHeaderDto = shinhanApiUtil.createHeaderDto(
                    "createDemandDepositAccount",
                    "createDemandDepositAccount",
                    apiKey,
                    TEMP_USER_KEY
            );

            ShinhanCreateDepositRequestHeaderDto headerDto = ShinhanCreateDepositRequestHeaderDto.builder()
                    .apiName(commonHeaderDto.getApiName())
                    .transmissionDate(commonHeaderDto.getTransmissionDate())
                    .transmissionTime(commonHeaderDto.getTransmissionTime())
                    .fintechAppNo(commonHeaderDto.getFintechAppNo())
                    .institutionCode(commonHeaderDto.getInstitutionCode())
                    .apiServiceCode(commonHeaderDto.getApiServiceCode())
                    .institutionTransactionUniqueNo(commonHeaderDto.getInstitutionTransactionUniqueNo())
                    .apiKey(commonHeaderDto.getApiKey())
                    .userKey(commonHeaderDto.getUserKey())
                    .build();

            ShinhanCreateDepositRequestDto requestDto = ShinhanCreateDepositRequestDto.builder()
                    .Header(headerDto)
                    .accountTypeUniqueNo(ACCOUNT_TYPE_UNIQUE_NO)
                    .build();

            logRequest(requestDto);

            ShinhanCreateDepositResponseDto response = shinhanApiClient.getClient("edu")
                    .post()
                    .uri("/demandDeposit/createDemandDepositAccount")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(ShinhanCreateDepositResponseDto.class)
                    .doOnNext(this::logResponse)
                    .block();

            return response;

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