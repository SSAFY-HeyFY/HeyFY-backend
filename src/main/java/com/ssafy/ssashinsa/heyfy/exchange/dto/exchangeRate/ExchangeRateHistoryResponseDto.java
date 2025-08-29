package com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "환율 그래프 단일 정보 dto")
public class ExchangeRateHistoryResponseDto {
    @Schema(description = "환율 통화", example = "USD")
    private String currency;
    private LocalDate date;
    @Schema(description = "환율", example = "1330.5")
    private double rate;
    @Schema(description = "예측 여부", example = "false")
    private boolean isPrediction;
    @Schema(description = "모델 이름")
    private String modelName;
}
