package com.ssafy.ssashinsa.heyfy.account.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InquireSingleTransactionHistoryRequestDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    private String accountNo;
    private String transactionUniqueNo;
}
