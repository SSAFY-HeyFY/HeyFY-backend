package com.ssafy.ssashinsa.heyfy.inquire.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonResponseHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanInquireDemandDepositResponseDto {
    @JsonProperty("Header")
    ShinhanCommonResponseHeaderDto Header;
    @JsonProperty("REC")
    private ShinhanInquireDepositResponseRecDto REC;
}