package com.ssafy.ssashinsa.heyfy.inquire.controller;

import com.ssafy.ssashinsa.heyfy.inquire.docs.GetForeignTransactionHistoryDocs;
import com.ssafy.ssashinsa.heyfy.inquire.docs.GetTransactionHistoryDocs;
import com.ssafy.ssashinsa.heyfy.account.dto.AccountNoDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.TransactionHistoryDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.TransactionHistoryResponseRecDto;
import com.ssafy.ssashinsa.heyfy.inquire.docs.InquireDepositListDocs;
import com.ssafy.ssashinsa.heyfy.inquire.docs.InquireSingleDepositDocs;
import com.ssafy.ssashinsa.heyfy.inquire.docs.InquireSingleForeignDepositDocs;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ForeignSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.SingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.inquire.service.InquireService;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history.InquireTransactionHistoryResponseRecDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireDepositResponseRecDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.text.DecimalFormat;
import java.util.List;
import java.util.stream.Collectors;

@Tag(name = "계좌/거래내역 조회", description = "신한은행 계좌 정보 조회 관련 API")
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



    @GetTransactionHistoryDocs
    @PostMapping("/transactionhistory")
    public ResponseEntity<TransactionHistoryResponseRecDto> getTransactionHistoryTest(@RequestBody AccountNoDto accountNo) {

        InquireTransactionHistoryResponseDto originalResponseDto = inquireService.getTransactionHistory(accountNo.getAccountNo());
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

        InquireTransactionHistoryResponseDto originalResponseDto = inquireService.getForeignTransactionHistory(accountNo.getAccountNo());
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
