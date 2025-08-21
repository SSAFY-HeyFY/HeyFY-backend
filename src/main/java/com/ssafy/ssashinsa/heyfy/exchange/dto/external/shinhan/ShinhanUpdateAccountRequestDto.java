package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ShinhanUpdateAccountRequestDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    private String accountNo;

    private Long transactionBalance;
    private double transactionSummary;
}
