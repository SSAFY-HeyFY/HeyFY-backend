package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonResponseHeaderDto;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ShinhanExchangeResponseRecDto {
    private ShinhanExchangeCurrencyRecDto exchangeCurrency;
    private AccountInfoRecDto accountInfo;
}
