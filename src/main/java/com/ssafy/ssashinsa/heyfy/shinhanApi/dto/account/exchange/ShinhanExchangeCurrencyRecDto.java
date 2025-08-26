package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.exchange;

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

