package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonResponseHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanInquireDepositResponseDto {
    @JsonProperty("Header")
    ShinhanCommonResponseHeaderDto Header;
    @JsonProperty("REC")
    private List<ShinhanInquireDepositResponseRecDto> REC = new ArrayList<>();
}