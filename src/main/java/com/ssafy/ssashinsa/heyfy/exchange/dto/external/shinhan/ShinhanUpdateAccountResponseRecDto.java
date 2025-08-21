package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ShinhanUpdateAccountResponseRecDto {
    private String transactionUniqueNo;
    private String transactionDate;
}
