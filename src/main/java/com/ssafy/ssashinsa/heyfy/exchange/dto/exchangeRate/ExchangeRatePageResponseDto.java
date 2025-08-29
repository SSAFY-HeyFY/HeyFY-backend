package com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "환율 페이지 응답 dto")
public class ExchangeRatePageResponseDto {
    @Schema(description = "환율 그래프 정보")
    private ExchangeRateHistoriesResponseDto exchangeRateHistories;
    @Schema(description = "실시간 환율 정보")
    private RealTimeRateGroupResponseDto realTimeRates;
    @Schema(description = "환율 예측 정보")
    private PredictionResponseDto prediction;
    @Schema(description = "환율 분석 정보")
    private TuitionResponseDto tuition;
}
