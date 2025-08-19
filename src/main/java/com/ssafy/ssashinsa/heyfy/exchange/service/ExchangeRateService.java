package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import lombok.extern.slf4j.Slf4j;
import com.ssafy.ssashinsa.heyfy.exchange.dto.EntireExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ExchangeRateRequestDto;
import org.apache.commons.lang3.RandomStringUtils;

@Slf4j
@Service
@RequiredArgsConstructor
public class ExchangeRateService {

    private final ShinhanApiClient apiClient;

    /**
     * Calls the external API to retrieve the entire exchange rate.
     *
     * @return EntireExchangeRateResponseDto if successful
     * @throws CustomException if API call fails
     */
    public EntireExchangeRateResponseDto getExchangeRate() {
        LocalDateTime now = LocalDateTime.now();
        String transmissionDate = now.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        String transmissionTime = now.format(DateTimeFormatter.ofPattern("HHmmss"));
        String institutionTransactionUniqueNo = now.format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) +
                RandomStringUtils.randomNumeric(6); // 6-digit random number;
        String apiKey = apiClient.getManagerKey();

        ShinhanCommonRequestHeaderDto headerDto = ShinhanCommonRequestHeaderDto.builder()
                .apiName("exchangeRate")
                .transmissionDate(transmissionDate)
                .transmissionTime(transmissionTime)
                .fintechAppNo("001")
                .institutionCode("00100")
                .apiServiceCode("exchangeRate")
                .institutionTransactionUniqueNo(institutionTransactionUniqueNo)
                .apiKey(apiKey)
                .build();
        ExchangeRateRequestDto requestDto = ExchangeRateRequestDto.builder()
                .Header(headerDto)
                .build();
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            log.info("Request JSON: {}", objectMapper.writeValueAsString(requestDto));
            return apiClient.getClient("exchange")
                    .post()
                    .uri("/exchangeRate")
                    .header("Content-Type", "application/json")
                    .bodyValue(requestDto)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, response ->
                            response.bodyToMono(String.class).flatMap(body -> {
                                log.error("API Error Body: {}", body);
                                return Mono.error(new CustomException(ErrorCode.API_CALL_FAILED));
                            }))
                    .bodyToMono(EntireExchangeRateResponseDto.class)
                    .doOnNext(res -> {
                        try {
                            log.info("Response JSON: {}", objectMapper.writeValueAsString(res));
                        } catch (Exception ex) {
                            log.error("Response logging error", ex);
                        }
                    })
                    .block();

        } catch (Exception e) {
            log.error("환율 조회 API 호출 실패: {}", e.getMessage(), e);
            throw new CustomException(ErrorCode.API_CALL_FAILED);
        }
    }
}