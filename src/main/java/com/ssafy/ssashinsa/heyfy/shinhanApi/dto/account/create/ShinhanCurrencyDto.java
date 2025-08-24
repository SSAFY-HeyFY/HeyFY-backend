package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanCurrencyDto {
    private String currency;
    private String currencyName;
}