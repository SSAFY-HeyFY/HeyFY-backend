package com.ssafy.ssashinsa.heyfy.exchange.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeRate30DaysResponseDto {
    // 3개 통화의 30일간 환율정보 리스트
    private java.util.List<Currency30DaysRatesDto> exchangeRateHistories;
    // usd, cny, vnd에 대한 단일 환율정보를 묶은 그룹
    private ExchangeRateGroupDto latestExchangeRate;
    private PredictionDto prediction;
    private TuitionDto tuition;
}
