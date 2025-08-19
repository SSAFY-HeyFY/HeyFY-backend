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
public class ShinhanCreateDepositRequestDto {
    @JsonProperty("Header")
    private ShinhanCreateDepositRequestHeaderDto Header;
    private String accountTypeUniqueNo;
}