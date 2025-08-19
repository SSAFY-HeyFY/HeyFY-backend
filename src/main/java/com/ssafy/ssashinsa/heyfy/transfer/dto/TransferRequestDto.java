package com.ssafy.ssashinsa.heyfy.transfer.dto;

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
public class TransferRequestDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;

    private String withdrawalAccountNo;
    private String depositAccountNo;
    private String transactionBalance;
}
