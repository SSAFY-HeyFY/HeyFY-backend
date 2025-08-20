package com.ssafy.ssashinsa.heyfy.account.controller;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountAuthResponseDto;
import com.ssafy.ssashinsa.heyfy.account.dto.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.account.service.AccountService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class AccountControllerTest { // 테스트용 컨트롤러

    private final AccountService accountService;

    @GetMapping("/accountauthtest")
    public ResponseEntity<AccountAuthResponseDto> getMyAccountAuthTest() {

        AccountAuthResponseDto accounts = accountService.AccountAuth();

        return ResponseEntity.ok(accounts);
    }

    @GetMapping("/transactionhistorytest")
    public ResponseEntity<InquireTransactionHistoryResponseDto> getTransactionHistoryTest() {

        InquireTransactionHistoryResponseDto transactionHistory = accountService.getTransactionHistory();

        return ResponseEntity.ok(transactionHistory);
    }
}