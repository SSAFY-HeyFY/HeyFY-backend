package com.ssafy.ssashinsa.heyfy.exchange.dto;

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
    private ShinhanCommonresponseHeaderDto Header;
    @JsonProperty("REC")
    private List<ExchangeRateResponseDto> REC = new ArrayList<>();
}
