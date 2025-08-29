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
@Schema
public class PredictionResponseDto {
    @Schema(description = "전망", example = "Bearish Trend")
    private String trend; // "bearish" | "bullish"
    private String description;
    @Schema(description = "변화량", example = "-3.5")
    private double changePercent;
    @Schema(description = "예측 기간(일)", example = "3")
    private int periodDays;
}

