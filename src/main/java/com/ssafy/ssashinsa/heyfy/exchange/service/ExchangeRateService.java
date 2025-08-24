package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeRate;
import com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate.*;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.EntireExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ExchangeRateRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan.ExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
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
    @Transactional
    public ExchangeRateHistoriesDto getExchangeRateHistories(String currencyCode, int day) {

        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(day);


        List<ExchangeRate> result =  exchangeRateRepository.findAllByCurrencyCodeAndBaseDateBetweenOrderByBaseDateAsc(
                currencyCode, startDate, endDate
        );
        List<ExchangeRateDto> rateDtos = result.stream()
                .map(rate -> ExchangeRateDto.builder()
                        .currency(rate.getCurrencyCode())
                        .date(rate.getBaseDate().toString())
                        .exchangeRate(rate.getRate().doubleValue())
                        .build())
                .toList();
        return ExchangeRateHistoriesDto.builder()
                .currency(currencyCode)
                .rates(rateDtos)
                .build();
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
                        .exchangeRate(parseExchangeRate(rec.getExchangeRate()))
                        .build();
            } else if ("CNY".equalsIgnoreCase(rec.getCurrency())) {
                cnyDto = ExchangeRateDto.builder()
                        .currency("CNY")
                        .date(rec.getCreatedAt())
                        .exchangeRate(parseExchangeRate(rec.getExchangeRate()))
                        .build();
            } else if ("VND".equalsIgnoreCase(rec.getCurrency())) {
                vndDto = ExchangeRateDto.builder()
                        .currency("VND")
                        .date(rec.getCreatedAt())
                        .exchangeRate(parseExchangeRate(rec.getExchangeRate()))
                        .build();
            }
        }
        ExchangeRateGroupDto latestExchangeRate = ExchangeRateGroupDto.builder()
                .usd(usdDto)
                .cny(cnyDto)
                .vnd(vndDto)
                .build();

        // 2. DB에서 30일간 USD 환율정보 조회
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(29);
        List<ExchangeRate> usdRates = exchangeRateRepository.findAllByCurrencyCodeAndBaseDateBetweenOrderByBaseDateAsc(
                "USD", startDate, endDate
        );
        List<ExchangeRateDto> usdRateDtos = usdRates.stream()
                .map(rate -> ExchangeRateDto.builder()
                        .currency(rate.getCurrencyCode())
                        .date(rate.getBaseDate().toString())
                        .exchangeRate(rate.getRate().doubleValue())
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

    /**
     * 외부 API에서 USD, CNY, VND 환율을 추출하여 반환
     */
    @Transactional
    public ExchangeRateGroupDto getCurrentExchangeRates() {
        EntireExchangeRateResponseDto apiResponse = getExchangeRateFromExternalApi();
        ExchangeRateDto usdDto = null;
        ExchangeRateDto cnyDto = null;
        ExchangeRateDto vndDto = null;
        for (ExchangeRateResponseDto rec : apiResponse.getREC()) {
            if ("USD".equalsIgnoreCase(rec.getCurrency())) {
                usdDto = ExchangeRateDto.builder()
                        .currency("USD")
                        .date(rec.getCreatedAt())
                        .exchangeRate(parseExchangeRate(rec.getExchangeRate()))
                        .build();
            } else if ("CNY".equalsIgnoreCase(rec.getCurrency())) {
                cnyDto = ExchangeRateDto.builder()
                        .currency("CNY")
                        .date(rec.getCreatedAt())
                        .exchangeRate(parseExchangeRate(rec.getExchangeRate()))
                        .build();
            } else if ("VND".equalsIgnoreCase(rec.getCurrency())) {
                vndDto = ExchangeRateDto.builder()
                        .currency("VND")
                        .date(rec.getCreatedAt())
                        .exchangeRate(parseExchangeRate(rec.getExchangeRate()))
                        .build();
            }
        }
        return ExchangeRateGroupDto.builder()
                .usd(usdDto)
                .cny(cnyDto)
                .vnd(vndDto)
                .build();
    }

    /**
     * 환율 예측 정보 반환 (더미 데이터)
     */
    @Transactional
    public PredictionDto getPrediction() {
        return PredictionDto.builder()
                .trend("bearish")
                .description("The rate might decline over the next 3 days")
                .changePercent(-1.24)
                .periodDays(3)
                .actionLabel("Exchange")
                .build();
    }

    /**
     * 학비 환율 추천 정보 반환 (더미 데이터)
     */
    @Transactional
    public TuitionDto getTuition() {
        return TuitionDto.builder()
                .period(PeriodDto.builder()
                        .start(java.time.LocalDate.of(2024, 3, 1))
                        .end(java.time.LocalDate.of(2024, 3, 31))
                        .build())
                .recommendedDate(java.time.LocalDate.of(2024, 3, 15))
                .recommendationNote("The exchange rate is expected to be highest on this day")
                .build();
    }

    private double parseExchangeRate(String rateStr) {
        if (rateStr == null) return 0.0;
        return Double.parseDouble(rateStr.replace(",", ""));
    }
}