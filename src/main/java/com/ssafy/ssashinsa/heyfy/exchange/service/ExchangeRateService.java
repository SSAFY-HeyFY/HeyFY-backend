package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import lombok.extern.slf4j.Slf4j;
import com.ssafy.ssashinsa.heyfy.exchange.dto.EntireExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ExchangeRateRequestDto;

@Slf4j
@Service
@RequiredArgsConstructor
public class ExchangeRateService {

    private final ShinhanApiClient apiClient;
    private final ShinhanApiUtil shinhanApiUtil;

    /**
     * Calls the external API to retrieve the entire exchange rate.
     *
     * @return EntireExchangeRateResponseDto if successful
     * @throws CustomException if API call fails
     */
    public EntireExchangeRateResponseDto getExchangeRate() {
        try {
            ExchangeRateRequestDto requestDto = createRequestDto();
            logRequest(requestDto);
            EntireExchangeRateResponseDto response = apiClient.getClient("edu")
                    .post()
                    .uri("/exchangeRate")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, r ->
                            r.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ShinhanApiErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(EntireExchangeRateResponseDto.class)
                    .doOnNext(this::logRequest)
                    .block();
            logResponse(response);
            return response;
        } catch (Exception e) {
            log.error("환율 조회 API 호출 실패: {}", e.getMessage(), e);
            throw new CustomException(ShinhanApiErrorCode.API_CALL_FAILED);
        }
    }

    private ExchangeRateRequestDto createRequestDto() {
        String apiKey = apiClient.getManagerKey();
        ShinhanCommonRequestHeaderDto headerDto = shinhanApiUtil.createHeaderDto("exchangeRate", "exchangeRate", apiKey, null);
        return ExchangeRateRequestDto.builder()
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