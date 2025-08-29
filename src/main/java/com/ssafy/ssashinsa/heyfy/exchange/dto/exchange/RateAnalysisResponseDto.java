package com.ssafy.ssashinsa.heyfy.exchange.dto.exchange;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "환율 분석 응답 dto")
public class RateAnalysisResponseDto {
    @Schema(description = "오늘 환율", example = "1330.5")
    private double todayRate;
    @Schema(description = "최종 예측 환율", example = "1340.2")
    private double finalPredictedRate;
    @Schema(description = "과거 분석 정보")
    private HistoricalAnalysisResponseDto historicalAnalysis;
    @Schema(description = "AI 예측 정보")
    private AIPredictionResponseDto aiPrediction;
}
