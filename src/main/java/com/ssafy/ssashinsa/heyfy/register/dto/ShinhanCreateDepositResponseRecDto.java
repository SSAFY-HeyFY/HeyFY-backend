package com.ssafy.ssashinsa.heyfy.register.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanCreateDepositResponseRecDto {
    private String bankCode;
    private String accountNo;
    private ShinhanCurrencyDto currency;
}