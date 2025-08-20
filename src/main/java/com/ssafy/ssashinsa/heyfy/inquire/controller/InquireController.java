package com.ssafy.ssashinsa.heyfy.inquire.controller;

import com.ssafy.ssashinsa.heyfy.inquire.dto.AccountCheckDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.inquire.service.InquireService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/inquire")
@RequiredArgsConstructor
public class InquireController {
    private final InquireService inquireService;

    @PostMapping("/depositlist")
    public ResponseEntity<ShinhanInquireDepositResponseDto> inquireDepositList() {

        ShinhanInquireDepositResponseDto response = inquireService.inquireDepositResponseDto();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/accountcheck")
    public ResponseEntity<AccountCheckDto> checkAccount() {
        boolean isAccountCheck = inquireService.checkAccount();
        AccountCheckDto response = new AccountCheckDto(isAccountCheck);

        return ResponseEntity.ok(response);
    }
}
