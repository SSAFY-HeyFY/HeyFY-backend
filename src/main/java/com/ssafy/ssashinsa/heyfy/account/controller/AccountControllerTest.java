package com.ssafy.ssashinsa.heyfy.account.controller;

import com.ssafy.ssashinsa.heyfy.account.service.AccountService;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthResponseDto;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@Tag(name = "계좌&거래내역 관리(백엔드 테스트용)", description = "계좌/거래내역 관리 API(백엔드 테스트용)")
public class AccountControllerTest { // 테스트용 컨트롤러

    private final AccountService accountService;

    @GetMapping("/accountauthtest")
    public ResponseEntity<AccountAuthResponseDto> getMyAccountAuthTest() {

        AccountAuthResponseDto accounts = accountService.AccountAuth();

        return ResponseEntity.ok(accounts);
    }

}