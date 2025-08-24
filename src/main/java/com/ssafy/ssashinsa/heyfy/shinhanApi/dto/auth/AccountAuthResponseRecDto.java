package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth;

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
