package com.ssafy.ssashinsa.heyfy.account.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.ShinhanCommonRequestHeaderDto;
import lombok.*;


@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AccountAuthRequestDto {
    @JsonProperty("Header")
    private ShinhanCommonRequestHeaderDto Header;
    private String accountNo;
    private String authText;
}
