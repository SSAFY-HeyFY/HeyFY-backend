package com.ssafy.ssashinsa.heyfy.inquire.controller;

import com.ssafy.ssashinsa.heyfy.inquire.service.InquireService;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import io.swagger.v3.oas.annotations.Hidden;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@Hidden
@RestController
@RequestMapping("/inquire")
@RequiredArgsConstructor
public class InquireControllerTest {
    private final InquireService inquireService;

    @Hidden
    @GetMapping("/singledeposittest") // 백엔드 테스트용 엔드포인트
    public ResponseEntity<ShinhanInquireSingleDepositResponseDto> inquireSingleDepositTest() {

        ShinhanInquireSingleDepositResponseDto response = inquireService.inquireSingleDeposit();

        return ResponseEntity.ok(response);
    }

    @Hidden
    @GetMapping("/singleforeigndeposittest") // 백엔드 테스트용 엔드포인트
    public ResponseEntity<ShinhanInquireSingleDepositResponseDto> inquireForeignSingleDepositTest() {

        ShinhanInquireSingleDepositResponseDto response = inquireService.inquireSingleForeignDeposit();

        return ResponseEntity.ok(response);
    }

    @Hidden
    @GetMapping("/transactionhistorytest")
    public ResponseEntity<InquireTransactionHistoryResponseDto> getTransactionHistoryTest() {

        InquireTransactionHistoryResponseDto transactionHistory = inquireService.getTransactionHistory();

        return ResponseEntity.ok(transactionHistory);
    }
}
