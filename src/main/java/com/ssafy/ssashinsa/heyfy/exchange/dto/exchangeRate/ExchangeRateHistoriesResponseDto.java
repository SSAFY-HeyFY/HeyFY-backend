package com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "환율 그래프 정보 dto")
public class ExchangeRateHistoriesResponseDto {
    @Schema(description = "환율 통화", example = "USD")
    private String currency; // USD, CNY, VND 등
    @Schema(description = "그래프 정보")
    private List<ExchangeRateHistoryResponseDto> rates; // 30일간 환율정보
}

