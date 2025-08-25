package com.ssafy.ssashinsa.heyfy.exchange.dto.exchangeRate;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeRateHistoryDto {
    private String currency;
    private LocalDate date;
    private double rate;
    private boolean isPrediction;
    private String modelName;
}
