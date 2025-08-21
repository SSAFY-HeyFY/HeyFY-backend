package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ShinhanExchangeCurrencyRecDto {
    private String amount;
    private String exchangeRate;
    private String currency;
    private String currencyName;
}

