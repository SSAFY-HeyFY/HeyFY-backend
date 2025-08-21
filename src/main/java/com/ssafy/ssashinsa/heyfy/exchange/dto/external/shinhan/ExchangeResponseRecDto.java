package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonResponseHeaderDto;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ExchangeResponseRecDto {
    @JsonProperty("Header")
    private ShinhanCommonResponseHeaderDto header;
    private ExchangeCurrencyRecDto exchangeCurrency;
    private AccountInfoRecDto accountInfo;
}
