package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanInquireSingleDepositRequestDto {
    @JsonProperty("Header")
    ShinhanCommonRequestHeaderDto Header;
    private String accountNo;
}