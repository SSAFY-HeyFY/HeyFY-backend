package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.common.ShinhanCommonRequestHeaderDto;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ShinhanUpdateAccountRequestDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    private String accountNo;

    private Double transactionBalance;
    private Double transactionSummary;
}
