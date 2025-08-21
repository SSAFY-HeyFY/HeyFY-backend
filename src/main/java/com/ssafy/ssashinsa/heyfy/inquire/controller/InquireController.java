package com.ssafy.ssashinsa.heyfy.inquire.controller;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountNoDto;
import com.ssafy.ssashinsa.heyfy.inquire.docs.CheckAccountDocs;
import com.ssafy.ssashinsa.heyfy.inquire.docs.InquireDepositListDocs;
import com.ssafy.ssashinsa.heyfy.inquire.docs.InquireSingleDepositDocs;
import com.ssafy.ssashinsa.heyfy.inquire.docs.InquireSingleForeignDepositDocs;
import com.ssafy.ssashinsa.heyfy.inquire.dto.*;
import com.ssafy.ssashinsa.heyfy.inquire.service.InquireService;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.text.DecimalFormat;
import java.util.List;

@Tag(name = "예금주 계좌 조회", description = "신한은행 계좌 정보 조회 관련 API")
@RestController
@RequestMapping("/inquire")
@RequiredArgsConstructor
public class InquireController {
    private final InquireService inquireService;

    @InquireDepositListDocs
    @GetMapping("/depositlist")
    public ResponseEntity<List<ShinhanInquireDepositResponseRecDto>> inquireDepositList() {

        ShinhanInquireDepositResponseDto response = inquireService.inquireDepositList();

        List<ShinhanInquireDepositResponseRecDto> depositList = response.getREC();

        return ResponseEntity.ok(depositList);
    }

    @InquireSingleDepositDocs
    @PostMapping("/singledeposit")
    public ResponseEntity<SingleDepositResponseDto> inquireSingleDeposit(@RequestBody AccountNoDto accountNo) {

        ShinhanInquireSingleDepositResponseDto response = inquireService.inquireSingleDeposit(accountNo.getAccountNo());
        ShinhanInquireDepositResponseRecDto rec = response.getREC();
        String accountBalanceString = String.valueOf((int) rec.getAccountBalance());
        SingleDepositResponseDto finalResponse = SingleDepositResponseDto.builder()
                .bankName(rec.getBankName())
                .userName(rec.getUserName())
                .accountNo(rec.getAccountNo())
                .accountName(rec.getAccountName())
                .accountBalance(accountBalanceString)
                .currency(rec.getCurrency())
                .build();

        return ResponseEntity.ok(finalResponse);
    }

    @InquireSingleForeignDepositDocs
    @PostMapping("/singleforeigndeposit")
    public ResponseEntity<ForeignSingleDepositResponseDto> inquireForeignSingleDeposit(@RequestBody AccountNoDto accountNo) {

        ShinhanInquireSingleDepositResponseDto response = inquireService.inquireSingleForeignDeposit(accountNo.getAccountNo());
        ShinhanInquireDepositResponseRecDto rec = response.getREC();
        DecimalFormat df = new DecimalFormat("0.00");
        String accountBalanceString = df.format(rec.getAccountBalance());

        ForeignSingleDepositResponseDto finalResponse = ForeignSingleDepositResponseDto.builder()
                .bankName(rec.getBankName())
                .userName(rec.getUserName())
                .accountNo(rec.getAccountNo())
                .accountName(rec.getAccountName())
                .accountBalance(accountBalanceString)
                .currency(rec.getCurrency())
                .build();

        return ResponseEntity.ok(finalResponse);
    }

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

    @CheckAccountDocs
    @GetMapping("/accountcheck")
    public ResponseEntity<AccountCheckDto> checkAccount() {
        boolean isAccountCheck = inquireService.checkAccount();
        AccountCheckDto response = new AccountCheckDto(isAccountCheck);

        return ResponseEntity.ok(response);
    }


}
