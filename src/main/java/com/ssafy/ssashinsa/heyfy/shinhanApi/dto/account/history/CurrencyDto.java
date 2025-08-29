package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CurrencyDto {

    @JsonProperty("currency")
    private String currency;

    @JsonProperty("currencyName")
    private String currencyName;

    @JsonProperty("amount")
    private Double amount;

}
