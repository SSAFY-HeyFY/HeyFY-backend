package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ShinhanExchangeRequestDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    private String accountNo;
    @Schema(description = "환전할 통화 코드, 예: USD, EUR 등")
    private String exchangeCurrent;
    @Schema(description = "환전할 금액(환전할 통화 기준)")
    private double exchangeAmount;
}
