package com.ssafy.ssashinsa.heyfy.fastapi.dto;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class FastApiRateAnalysisDto {
    private LocalDateTime apiCalledAt;
    private double todayRate;
    private double aiPredictedRate;
    private String historicalAnalysis;
    private String aiPrediction;
}

