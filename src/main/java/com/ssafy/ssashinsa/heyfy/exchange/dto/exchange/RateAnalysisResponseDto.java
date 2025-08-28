package com.ssafy.ssashinsa.heyfy.exchange.dto.exchange;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RateAnalysisResponseDto {
    private double todayRate;
    private double finalPredictedRate;
    private HistoricalAnalysisResponseDto historicalAnalysis;
    private AIPredictionResponseDto aiPrediction;
}
