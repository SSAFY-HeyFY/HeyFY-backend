package com.ssafy.ssashinsa.heyfy.exchange.dto.exchange;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ExchangeResponseDto {
    private Double depositAccountBalance;
    private Double withdrawalAccountBalance;
    private Double transactionBalance;
}
