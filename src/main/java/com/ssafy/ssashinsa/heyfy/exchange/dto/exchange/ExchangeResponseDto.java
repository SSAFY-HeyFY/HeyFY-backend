package com.ssafy.ssashinsa.heyfy.exchange.dto.exchange;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ExchangeResponseDto {
    private Double depositAccountBalance;
    private Double withdrawalAccountBalance;
    private Double transactionBalance;
    @JsonProperty("isCorrect")
    private boolean isCorrect;
}
