package com.ssafy.ssashinsa.heyfy.account.controller;

import com.ssafy.ssashinsa.heyfy.account.docs.GetMyAccountAuthDocs;
import com.ssafy.ssashinsa.heyfy.account.docs.GetMyAccountsDocs;
import com.ssafy.ssashinsa.heyfy.account.docs.GetTransactionHistoryDocs;
import com.ssafy.ssashinsa.heyfy.account.dto.*;
import com.ssafy.ssashinsa.heyfy.account.service.AccountService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
@RequiredArgsConstructor
public class AccountController {

    private final AccountService accountService;

    @GetMyAccountsDocs
    @GetMapping("/accounts")
    public ResponseEntity<AccountPairDto> getMyAccounts() {

        Optional<AccountPairDto> accounts = accountService.getAccounts();

        return accounts.map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @GetMyAccountAuthDocs
    @PostMapping("/accountauth")
    public ResponseEntity<AccountAuthHttpResponseDto> getMyAccountAuth() {
        AccountAuthResponseDto accountAuthResponse = accountService.AccountAuth();

        String message = "1원 계좌 인증에 성공했습니다.";
        String accountNo = accountAuthResponse.getREC().getAccountNo();
        AccountAuthHttpResponseDto responseDto = new AccountAuthHttpResponseDto(message, accountNo);

        return ResponseEntity.ok(responseDto);
    }

    @GetTransactionHistoryDocs
    @GetMapping("/transactionhistory")
    public ResponseEntity<InquireTransactionHistoryResponseRecDto> getTransactionHistoryTest() {

        InquireTransactionHistoryResponseDto transactionHistoryDto = accountService.getTransactionHistory();

        InquireTransactionHistoryResponseRecDto rec = transactionHistoryDto.getREC();

        return ResponseEntity.ok(rec);
    }
}