package com.ssafy.ssashinsa.heyfy.exchange.service;

import com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate.*;
import com.ssafy.ssashinsa.heyfy.exchange.repository.ExchangeRateRepository;
import com.ssafy.ssashinsa.heyfy.fastapi.client.FastApiClient;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRateGraphDto;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRealTimeRateDto;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.FastApiRealTimeRatesDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.config.ShinhanApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.utils.ShinhanApiUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
public class ExchangeRateService {

    private final ShinhanApiClient apiClient;
    private final ShinhanApiUtil shinhanApiUtil;
    private final ExchangeRateRepository exchangeRateRepository;
    private final FastApiClient fastApiClient;

    /**
     * 특정 통화의 최근 30일 환율 조회
     */
    @Transactional
    public ExchangeRateHistoriesResponseDto getExchangeRateHistories() {

        FastApiRateGraphDto apiResponse = fastApiClient.getRateGraph();

        List<ExchangeRateHistoryResponseDto> rateDtos = apiResponse.getData().stream()
                .map(rate -> ExchangeRateHistoryResponseDto.builder()
                        .currency("USD")
                        .date(rate.getDate())
                        .rate(rate.getRate())
                        .isPrediction(rate.isPrediction())
                        .modelName(rate.isPrediction() ? rate.getModelName() : "")
                        .build())
                .toList();
        return ExchangeRateHistoriesResponseDto.builder()
                .currency("USD")
                .rates(rateDtos)
                .build();
    }

    @Transactional
    public ExchangeRatePageResponseDto getExchangeRatePage() {
        // 1. ExchangeRateHistories API 호출
        FastApiRateGraphDto rateGraphResponse = fastApiClient.getRateGraph();

        List<ExchangeRateHistoryResponseDto> rateDtos = rateGraphResponse.getData().stream()
                .map(rate -> ExchangeRateHistoryResponseDto.builder()
                        .currency("USD")
                        .date(rate.getDate())
                        .rate(rate.getRate())
                        .isPrediction(rate.isPrediction())
                        .modelName(rate.isPrediction() ? rate.getModelName() : "")
                        .build())
                .toList();
        ExchangeRateHistoriesResponseDto exchangeRateHistories = ExchangeRateHistoriesResponseDto.builder()
                .currency("USD")
                .rates(rateDtos)
                .build();
        // 2. RealTimeRates API 호출
        FastApiRealTimeRatesDto realTimeRatesResponse = fastApiClient.getRealTimeRates();
        RealTimeRateResponseDto usdDto = null;
        RealTimeRateResponseDto cnyDto = null;
        RealTimeRateResponseDto vndDto = null;
        for (FastApiRealTimeRateDto data : realTimeRatesResponse.getData()) {
            if ("USD".equalsIgnoreCase(data.getCurrency())) {
                usdDto = RealTimeRateResponseDto.builder()
                        .currency("USD")
                        .updatedAt(data.getUpdatedAt())
                        .rate(data.getRate())
                        .build();
            } else if ("CNY".equalsIgnoreCase(data.getCurrency())) {
                cnyDto = RealTimeRateResponseDto.builder()
                        .currency("CNY")
                        .updatedAt(data.getUpdatedAt())
                        .rate(data.getRate())
                        .build();
            } else if ("VND".equalsIgnoreCase(data.getCurrency())) {
                vndDto = RealTimeRateResponseDto.builder()
                        .currency("VND")
                        .updatedAt(data.getUpdatedAt())
                        .rate(data.getRate())
                        .build();
            }
        }
        RealTimeRateGroupResponseDto realTimeRateGroup = RealTimeRateGroupResponseDto.builder()
                .usd(usdDto)
                .cny(cnyDto)
                .vnd(vndDto)
                .build();

        // 3. Prediction, Tuition 더미 데이터 생성
        PredictionResponseDto prediction = PredictionResponseDto.builder()
                .trend("bearish")
                .description("The rate might decline over the next 3 days")
                .changePercent(-1.24)
                .periodDays(3)
                .actionLabel("Exchange")
                .build();
        TuitionResponseDto tuition = TuitionResponseDto.builder()
                .period(PeriodResponseDto.builder()
                        .start(java.time.LocalDate.of(2024, 3, 1))
                        .end(java.time.LocalDate.of(2024, 3, 31))
                        .build())
                .recommendedDate(java.time.LocalDate.of(2024, 3, 15))
                .recommendationNote("The exchange rate is expected to be highest on this day")
                .build();

        // 4. 최종 DTO 조립
        return ExchangeRatePageResponseDto.builder()
                .exchangeRateHistories(exchangeRateHistories)
                .realTimeRates(realTimeRateGroup)
                .prediction(prediction)
                .tuition(tuition)
                .build();
    }

    /**
     * 외부 API에서 USD, CNY, VND 환율을 추출하여 반환
     */
    @Transactional
    public RealTimeRateGroupResponseDto getRealTimeRate() {
        FastApiRealTimeRatesDto apiResponse = fastApiClient.getRealTimeRates();
        RealTimeRateResponseDto usdDto = null;
        RealTimeRateResponseDto cnyDto = null;
        RealTimeRateResponseDto vndDto = null;
        for (FastApiRealTimeRateDto data : apiResponse.getData()) {
            if ("USD".equalsIgnoreCase(data.getCurrency())) {
                usdDto = RealTimeRateResponseDto.builder()
                        .currency("USD")
                        .updatedAt(data.getUpdatedAt())
                        .rate(data.getRate())
                        .build();
            } else if ("CNY".equalsIgnoreCase(data.getCurrency())) {
                cnyDto = RealTimeRateResponseDto.builder()
                        .currency("CNY")
                        .updatedAt(data.getUpdatedAt())
                        .rate(data.getRate())
                        .build();
            } else if ("VND".equalsIgnoreCase(data.getCurrency())) {
                vndDto = RealTimeRateResponseDto.builder()
                        .currency("VND")
                        .updatedAt(data.getUpdatedAt())
                        .rate(data.getRate())
                        .build();
            }
        }
        return RealTimeRateGroupResponseDto.builder()
                .usd(usdDto)
                .cny(cnyDto)
                .vnd(vndDto)
                .build();
    }

    /**
     * 환율 예측 정보 반환 (더미 데이터)
     */
    @Transactional
    public PredictionResponseDto getPrediction() {
        return PredictionResponseDto.builder()
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
    public TuitionResponseDto getTuition() {
        return TuitionResponseDto.builder()
                .period(PeriodResponseDto.builder()
                        .start(java.time.LocalDate.of(2024, 3, 1))
                        .end(java.time.LocalDate.of(2024, 3, 31))
                        .build())
                .recommendedDate(java.time.LocalDate.of(2024, 3, 15))
                .recommendationNote("The exchange rate is expected to be highest on this day")
                .build();
    }
}