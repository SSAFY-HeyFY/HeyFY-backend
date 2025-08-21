package com.ssafy.ssashinsa.heyfy.home.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class HomeDto {
    private String studentId;
    private AccountInfo normalAccount;
    private AccountInfo foreignAccount;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AccountInfo {
        private String accountNo;
        private long balance;
        private String currency; // 외화 계좌용
    }
}
