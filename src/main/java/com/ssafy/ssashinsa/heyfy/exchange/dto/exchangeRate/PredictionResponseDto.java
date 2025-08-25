package com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionResponseDto {
    private String trend; // "bearish" | "bullish"
    private String description;
    private double changePercent;
    private int periodDays;
    private String actionLabel;
}

