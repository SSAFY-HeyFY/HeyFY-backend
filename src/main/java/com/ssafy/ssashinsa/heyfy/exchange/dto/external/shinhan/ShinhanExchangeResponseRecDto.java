package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ShinhanExchangeResponseRecDto {
    private ShinhanExchangeCurrencyRecDto exchangeCurrency;
    private AccountInfoRecDto accountInfo;
}
