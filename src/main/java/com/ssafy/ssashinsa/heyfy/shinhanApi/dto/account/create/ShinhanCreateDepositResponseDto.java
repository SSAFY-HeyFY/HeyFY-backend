package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonResponseHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanCreateDepositResponseDto {
    @JsonProperty("Header")
    private ShinhanCommonResponseHeaderDto Header;
    @JsonProperty("REC")
    private ShinhanCreateDepositResponseRecDto REC;
}