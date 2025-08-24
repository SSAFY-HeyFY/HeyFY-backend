package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

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
public class ShinhanInquireDemandDepositAccountBalanceResponseDto {
    @JsonProperty("Header")
    ShinhanCommonResponseHeaderDto Header;
    @JsonProperty("REC")
    private ShinhanInquireDemandDepositAccountBalanceResponseRecDto REC;
}
