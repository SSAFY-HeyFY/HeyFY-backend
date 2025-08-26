package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;

public class ShinhanUpdateAccountResponseDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    @JsonProperty("REC")
    private ShinhanUpdateAccountResponseRecDto REC;
}
