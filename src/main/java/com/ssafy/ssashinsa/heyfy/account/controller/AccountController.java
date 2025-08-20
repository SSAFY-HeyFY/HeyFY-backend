package com.ssafy.ssashinsa.heyfy.account.controller;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountPairDto;
import com.ssafy.ssashinsa.heyfy.account.dto.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.account.service.AccountService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@RestController
@RequiredArgsConstructor
public class AccountController {

    private final AccountService accountService;

    @GetMapping("/accounts")
    public ResponseEntity<AccountPairDto> getMyAccounts() {

        Optional<AccountPairDto> accounts = accountService.getAccounts();

        return accounts.map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @GetMapping("/accountauth")
    public ResponseEntity<Map<String, String>> getMyAccountAuth() { // 프론트엔드 구현을 위한 로직
        accountService.AccountAuth();

        Map<String, String> response = new HashMap<>();
        response.put("message", "1원 계좌 인증에 성공했습니다.");
        return ResponseEntity.ok(response);

    }

    @GetMapping("/transactionhistory")
    public ResponseEntity<InquireTransactionHistoryResponseDto> getTransactionHistoryTest() {

        InquireTransactionHistoryResponseDto transactionHistory = accountService.getTransactionHistory();

        return ResponseEntity.ok(transactionHistory);
    }
}