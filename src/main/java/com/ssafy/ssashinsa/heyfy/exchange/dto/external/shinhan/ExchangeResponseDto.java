package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
@Builder
public class ExchangeResponseDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    @JsonProperty("REC")
    private ExchangeResponseRecDto REC;
}
