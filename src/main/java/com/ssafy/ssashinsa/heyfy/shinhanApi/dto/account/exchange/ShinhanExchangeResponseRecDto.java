package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.exchange;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class ShinhanExchangeResponseRecDto {
    private ShinhanExchangeCurrencyRecDto exchangeCurrency;
    private ShinhanAccountInfoRecDto accountInfo;
}
