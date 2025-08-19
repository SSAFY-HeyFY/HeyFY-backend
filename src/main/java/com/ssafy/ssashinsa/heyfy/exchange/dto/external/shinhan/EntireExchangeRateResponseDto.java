package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonResponseHeaderDto;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class EntireExchangeRateResponseDto {

    @JsonProperty("Header")
    private ShinhanCommonResponseHeaderDto Header;
    @JsonProperty("REC")
    private List<ExchangeRateResponseDto> REC = new ArrayList<>();
}
