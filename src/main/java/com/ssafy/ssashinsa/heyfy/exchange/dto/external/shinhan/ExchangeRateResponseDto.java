package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ExchangeRateResponseDto {
    private Long id;
    private String currency;
    private String exchangeRate;
    private String exchangeMin;
    @JsonProperty("created")
    private String createdAt;
}

