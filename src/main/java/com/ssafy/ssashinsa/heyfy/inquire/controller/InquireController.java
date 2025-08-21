package com.ssafy.ssashinsa.heyfy.inquire.controller;

import com.ssafy.ssashinsa.heyfy.inquire.docs.CheckAccountDocs;
import com.ssafy.ssashinsa.heyfy.inquire.docs.InquireDepositListDocs;
import com.ssafy.ssashinsa.heyfy.inquire.docs.InquireSingleDepositDocs;
import com.ssafy.ssashinsa.heyfy.inquire.dto.AccountCheckDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireDepositResponseRecDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.inquire.service.InquireService;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

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
    @GetMapping("/singledeposit")
    public ResponseEntity<ShinhanInquireDepositResponseRecDto> inquireSingleDeposit() {

        ShinhanInquireSingleDepositResponseDto response = inquireService.inquireSingleDeposit();

        return ResponseEntity.ok(response.getREC());
    }

    @GetMapping("/singleforeigndeposit")
    public ResponseEntity<ShinhanInquireDepositResponseRecDto> inquireForeignSingleDeposit() {

        ShinhanInquireSingleDepositResponseDto response = inquireService.inquireSingleDeposit();

        return ResponseEntity.ok(response.getREC());
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
