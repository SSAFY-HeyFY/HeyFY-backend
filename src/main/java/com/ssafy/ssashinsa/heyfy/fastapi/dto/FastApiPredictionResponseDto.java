package com.ssafy.ssashinsa.heyfy.fastapi.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FastApiPredictionResponseDto {
    @JsonProperty("trend_type")
    private String trendType;
    @JsonProperty("header")
    private String header;
    @JsonProperty("change_label")
    private FastApiPredictionSummaryRateDto changeLabel;
    @JsonProperty("highlight_date")
    private String highlightDate;
    @JsonProperty("highlight_rate")
    private double highlightRate;
}