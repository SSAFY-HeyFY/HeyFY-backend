package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeRate;
import com.ssafy.ssashinsa.heyfy.exchange.dto.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.EntireExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ExchangeRateRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ShinhanCommonRequestHeaderDto;
import com.ssafy.ssashinsa.heyfy.exchange.repository.ExchangeRateRepository;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import org.springframework.transaction.annotation.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.LocalDate;
import java.util.List;

@Slf4j
@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class ExchangeRateService {

    private final ShinhanApiClient apiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ExchangeRateRepository exchangeRateRepository;

    /**
     * 특정 통화의 최근 30일 환율 조회
     */
    public List<ExchangeRate> getLast30DaysRates(String currencyCode) {
        LocalDate endDate = LocalDate.now().minusDays(1); // 오늘은 외부 API로 처리하므로 제외
        LocalDate startDate = endDate.minusDays(29);      // 총 30일치 (end 포함)

        List<ExchangeRate> result =  exchangeRateRepository.findAllByCurrencyCodeAndBaseDateBetweenOrderByBaseDateAsc(
                currencyCode, startDate, endDate
        );

        return result;
    }
    /**
     * Calls the external API to retrieve the entire exchange rate.
     *
     * @return EntireExchangeRateResponseDto if successful
     * @throws CustomException if API call fails
     */
    public EntireExchangeRateResponseDto getExchangeRateFromExternalApi() {
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

    @Transactional
    public ExchangeRatePageResponseDto getExchangeRatePage() {
        // 1. 외부 API에서 최신 환율정보 추출
        EntireExchangeRateResponseDto apiResponse = getExchangeRateFromExternalApi();
        ExchangeRateDto usdDto = null;
        ExchangeRateDto cnyDto = null;
        ExchangeRateDto vndDto = null;
        for (ExchangeRateResponseDto rec : apiResponse.getREC()) {
            if ("USD".equalsIgnoreCase(rec.getCurrency())) {
                usdDto = ExchangeRateDto.builder()
                        .currency("USD")
                        .date(rec.getCreatedAt())
                        .exchangeRate(rec.getExchangeRate())
                        .build();
            } else if ("CNY".equalsIgnoreCase(rec.getCurrency())) {
                cnyDto = ExchangeRateDto.builder()
                        .currency("CNY")
                        .date(rec.getCreatedAt())
                        .exchangeRate(rec.getExchangeRate())
                        .build();
            } else if ("VND".equalsIgnoreCase(rec.getCurrency())) {
                vndDto = ExchangeRateDto.builder()
                        .currency("VND")
                        .date(rec.getCreatedAt())
                        .exchangeRate(rec.getExchangeRate())
                        .build();
            }
        }
        ExchangeRateGroupDto latestExchangeRate = ExchangeRateGroupDto.builder()
                .usd(usdDto)
                .cny(cnyDto)
                .vnd(vndDto)
                .build();

        // 2. DB에서 30일간 USD 환율정보 조회
        List<ExchangeRate> usdRates = getLast30DaysRates("USD");
        List<ExchangeRateDto> usdRateDtos = usdRates.stream()
                .map(rate -> ExchangeRateDto.builder()
                        .currency(rate.getCurrencyCode())
                        .date(rate.getBaseDate().toString())
                        .exchangeRate(rate.getRate().toString())
                        .build())
                .toList();
        ExchangeRateHistoriesDto exchangeRateHistories = ExchangeRateHistoriesDto.builder()
                .currency("USD")
                .rates(usdRateDtos)
                .build();

        // 3. Prediction, Tuition 더미 데이터 생성
        PredictionDto prediction = PredictionDto.builder()
                .trend("bearish")
                .description("The rate might decline over the next 3 days")
                .changePercent(-1.24)
                .periodDays(3)
                .actionLabel("Exchange")
                .build();
        TuitionDto tuition = TuitionDto.builder()
                .period(PeriodDto.builder()
                        .start(java.time.LocalDate.of(2024, 3, 1))
                        .end(java.time.LocalDate.of(2024, 3, 31))
                        .build())
                .recommendedDate(java.time.LocalDate.of(2024, 3, 15))
                .recommendationNote("The exchange rate is expected to be highest on this day")
                .build();

        // 4. 최종 DTO 조립
        return ExchangeRatePageResponseDto.builder()
                .exchangeRateHistories(exchangeRateHistories)
                .latestExchangeRate(latestExchangeRate)
                .prediction(prediction)
                .tuition(tuition)
                .build();
    }
}