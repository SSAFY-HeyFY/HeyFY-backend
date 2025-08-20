package com.ssafy.ssashinsa.heyfy.account.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AccountAuthResponseRecDto {
    private String transactionUniqueNo;
    private String accountNo;

}
