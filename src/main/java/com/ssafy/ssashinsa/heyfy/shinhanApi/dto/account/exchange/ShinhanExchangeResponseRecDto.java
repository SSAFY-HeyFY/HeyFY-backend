package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.exchange;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ShinhanExchangeResponseRecDto {
    private ShinhanExchangeCurrencyRecDto exchangeCurrency;
    private ShinhanAccountInfoRecDto accountInfo;
}
