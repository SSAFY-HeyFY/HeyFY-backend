package com.ssafy.ssashinsa.heyfy.inquire.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.exchange.dto.ShinhanCommonRequestHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanInquireDepositRequestDto {
    @JsonProperty("Header")
    ShinhanCommonRequestHeaderDto Header;
}