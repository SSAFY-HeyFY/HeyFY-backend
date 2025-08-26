package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.foreign;

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
public class ShinhanCreateforeignDepositRequestDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    private String accountTypeUniqueNo;
    private String currency;
}