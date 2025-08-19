package com.ssafy.ssashinsa.heyfy.transfer.dto;

import com.ssafy.ssashinsa.heyfy.exchange.dto.ExchangeRateResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ShinhanCommonResponseHeaderDto;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class EntireTransferResponseDto {
    @JsonProperty("Header")
    private ShinhanCommonResponseHeaderDto Header;
    @JsonProperty("REC")
    private List<TransferResponseDto> REC = new ArrayList<>();
}
