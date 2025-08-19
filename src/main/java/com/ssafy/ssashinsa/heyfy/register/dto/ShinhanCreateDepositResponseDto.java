package com.ssafy.ssashinsa.heyfy.register.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
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
    private ShinhanCreateDepositResponseHeaderDto Header;
    @JsonProperty("REC")
    private ShinhanCreateDepositResponseRecDto REC;
}