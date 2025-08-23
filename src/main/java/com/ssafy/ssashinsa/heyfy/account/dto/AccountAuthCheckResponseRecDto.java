package com.ssafy.ssashinsa.heyfy.account.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AccountAuthCheckResponseRecDto {
    private String status;
    private String transactionUniqueNo;
    private String accountNo;
}
