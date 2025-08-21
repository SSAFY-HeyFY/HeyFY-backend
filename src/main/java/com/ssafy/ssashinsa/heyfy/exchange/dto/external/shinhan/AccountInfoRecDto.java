package com.ssafy.ssashinsa.heyfy.exchange.dto.external.shinhan;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AccountInfoRecDto {
    private String accountNo;
    private String amount;
    private String balance;
}
