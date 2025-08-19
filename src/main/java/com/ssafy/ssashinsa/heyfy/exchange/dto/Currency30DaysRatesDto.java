package com.ssafy.ssashinsa.heyfy.exchange.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Currency30DaysRatesDto {
    private String currency; // USD, CNY, VND 등
    private List<ExchangeRateDto> rates; // 30일간 환율정보
}

