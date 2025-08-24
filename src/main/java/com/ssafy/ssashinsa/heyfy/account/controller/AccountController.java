package com.ssafy.ssashinsa.heyfy.account.controller;

import com.ssafy.ssashinsa.heyfy.account.docs.*;
import com.ssafy.ssashinsa.heyfy.account.dto.*;
import com.ssafy.ssashinsa.heyfy.account.service.AccountService;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.register.service.RegisterService;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireSingleTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseRecDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthCheckResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.auth.AccountAuthResponseDto;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.text.DecimalFormat;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@RestController
@RequiredArgsConstructor
@Tag(name = "계좌/거래내역 관리", description = "계좌/거래내역 관리 API")
public class AccountController {

    private final AccountService accountService;
    private final RegisterService registerService;

    @GetMyAccountsDocs
    @GetMapping("/accounts")
    public ResponseEntity<AccountPairDto> getMyAccounts() {

        Optional<AccountPairDto> accounts = accountService.getAccounts();

        return accounts.map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @GetMyAccountAuthDocs
    @PostMapping("/accountauth")
    public ResponseEntity<AccountAuthHttpResponseDto> getMyAccountAuth(@RequestBody AccountNoDto accountNoDto) {
        String accountNo = accountNoDto.getAccountNo();
        AccountAuthResponseDto accountAuthResponse = accountService.AccountAuth(accountNo);

        InquireSingleTransactionHistoryResponseDto singleTransactionHistoryResponse = accountService.getSingleTransactionHistory(accountNo, accountAuthResponse.getREC().getTransactionUniqueNo());

        String message = singleTransactionHistoryResponse.getREC().getTransactionSummary();
        String[] parts = message.split(" ");
        String lastFour = parts[parts.length - 1];
        AccountAuthHttpResponseDto responseDto = new AccountAuthHttpResponseDto(lastFour, accountNo);

        return ResponseEntity.ok(responseDto);
    }

    @AccountCheckDocs
    @PostMapping("/accouncheck")
    public ResponseEntity<AccountNoDto> AccountCheck(@RequestBody AuthCheckDto authCheckDto) {

        try {
            AccountAuthCheckResponseDto accountAuthCheckResponse = accountService.accountAuthCheck(authCheckDto.getAccountNo(), authCheckDto.getAuthCode());

            // 인증 완료시, db상에 일반 계좌 등록.
            registerService.registerAccount(accountAuthCheckResponse.getREC().getAccountNo());

            AccountNoDto AccountNoDto = new AccountNoDto(accountAuthCheckResponse.getREC().getAccountNo());

            return ResponseEntity.ok(AccountNoDto);
        } catch (CustomException e) {
            if (e.getErrorCode() == ShinhanRegisterApiErrorCode.API_CALL_FAILED) {
                throw new CustomException(ShinhanRegisterApiErrorCode.FAIL_CHECK_AUTH);
            }
            throw e;
        }



    }

    @GetTransactionHistoryDocs
    @PostMapping("/transactionhistory")
    public ResponseEntity<TransactionHistoryResponseRecDto> getTransactionHistoryTest(@RequestBody AccountNoDto accountNo) {

        InquireTransactionHistoryResponseDto originalResponseDto = accountService.getTransactionHistory(accountNo.getAccountNo());
        InquireTransactionHistoryResponseRecDto originalRec = originalResponseDto.getREC();

        List<TransactionHistoryDto> newList = originalRec.getList().stream()
                .map(originalItem -> TransactionHistoryDto.builder()
                        .transactionUniqueNo(originalItem.getTransactionUniqueNo())
                        .transactionDate(originalItem.getTransactionDate())
                        .transactionTime(originalItem.getTransactionTime())
                        .transactionType(originalItem.getTransactionType())
                        .transactionTypeName(originalItem.getTransactionTypeName())
                        .transactionAccountNo(originalItem.getTransactionAccountNo())
                        .transactionBalance(String.valueOf((int) originalItem.getTransactionBalance()))
                        .transactionAfterBalance(String.valueOf((int) originalItem.getTransactionAfterBalance()))
                        .transactionSummary(originalItem.getTransactionSummary())
                        .transactionMemo(originalItem.getTransactionMemo())
                        .build())
                .collect(Collectors.toList());

        TransactionHistoryResponseRecDto finalResponse = TransactionHistoryResponseRecDto.builder()
                .totalCount(originalRec.getTotalCount())
                .list(newList)
                .build();

        return ResponseEntity.ok(finalResponse);
    }

    @GetForeignTransactionHistoryDocs
    @PostMapping("/foreigntransactionhistory")
    public ResponseEntity<TransactionHistoryResponseRecDto> getForeignTransactionHistoryTest(@RequestBody AccountNoDto accountNo) {

        InquireTransactionHistoryResponseDto originalResponseDto = accountService.getForeignTransactionHistory(accountNo.getAccountNo());
        InquireTransactionHistoryResponseRecDto originalRec = originalResponseDto.getREC();

        DecimalFormat df = new DecimalFormat("0.00");
        List<TransactionHistoryDto> newList = originalRec.getList().stream()
                .map(originalItem -> TransactionHistoryDto.builder()
                        .transactionUniqueNo(originalItem.getTransactionUniqueNo())
                        .transactionDate(originalItem.getTransactionDate())
                        .transactionTime(originalItem.getTransactionTime())
                        .transactionType(originalItem.getTransactionType())
                        .transactionTypeName(originalItem.getTransactionTypeName())
                        .transactionAccountNo(originalItem.getTransactionAccountNo())
                        .transactionBalance(df.format(originalItem.getTransactionBalance()))
                        .transactionAfterBalance(df.format(originalItem.getTransactionAfterBalance()))
                        .transactionSummary(originalItem.getTransactionSummary())
                        .transactionMemo(originalItem.getTransactionMemo())
                        .build())
                .collect(Collectors.toList());

        TransactionHistoryResponseRecDto finalResponse = TransactionHistoryResponseRecDto.builder()
                .totalCount(originalRec.getTotalCount())
                .list(newList)
                .build();

        return ResponseEntity.ok(finalResponse);
    }
}