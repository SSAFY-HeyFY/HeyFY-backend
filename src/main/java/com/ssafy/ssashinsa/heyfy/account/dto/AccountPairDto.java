package com.ssafy.ssashinsa.heyfy.account.dto;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class AccountPairDto {
    private Account account;
    private ForeignAccount foreignAccount;
}