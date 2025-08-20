package com.ssafy.ssashinsa.heyfy.transfer.controller;

import com.ssafy.ssashinsa.heyfy.account.exception.AccountErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.swagger.docs.ErrorsCommonDocs;
import com.ssafy.ssashinsa.heyfy.transfer.docs.TransferDocs;
import com.ssafy.ssashinsa.heyfy.transfer.dto.*;
import com.ssafy.ssashinsa.heyfy.transfer.service.TransferService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.OffsetDateTime;
import java.time.ZoneId;

@RestController
@Tag(name = "Transfer", description = "이체 관련 API")
@RequestMapping("/transfers")
@RequiredArgsConstructor
@ErrorsCommonDocs
public class TransferController {

    private final TransferService transferService;

    @PostMapping("/domestic")
    @TransferDocs
    public TransferHistoryResponse transfer(@RequestBody CreateTransferRequest req) {
        EntireTransferResponseDto transferResponse = transferService.callTransfer(
                req.getDepositAccountNo(), req.getAmount()
        );

        String withdrawalAccountNo = transferResponse.getREC().get(0).getAccountNo();
        if (withdrawalAccountNo == null || withdrawalAccountNo.isEmpty()) {
            throw new CustomException(AccountErrorCode.WITHDRAWAL_ACCOUNT_NOT_FOUND); // 계좌 없을 시 예외 처리
        }

        TransferHistory history = new TransferHistory(
                withdrawalAccountNo,
                req.getDepositAccountNo(),
                req.getAmount(),
                "KRW",
                OffsetDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return TransferHistoryResponse.ok(history);
    }

    @PostMapping("/foreign")
    @TransferDocs
    public TransferHistoryResponse foreignTransfer(@RequestBody CreateTransferRequest req) {
        EntireTransferResponseDto transferResponse = transferService.callForeignTransfer(
                req.getDepositAccountNo(), req.getAmount()
        );

        String withdrawalAccountNo = transferResponse.getREC().get(0).getAccountNo();
        if (withdrawalAccountNo == null || withdrawalAccountNo.isEmpty()) {
            throw new CustomException(AccountErrorCode.WITHDRAWAL_ACCOUNT_NOT_FOUND); // 계좌 없을 시 예외 처리
        }

        TransferHistory history = new TransferHistory(
                withdrawalAccountNo,
                req.getDepositAccountNo(),
                req.getAmount(),
                "USD",
                OffsetDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return TransferHistoryResponse.ok(history);
    }
}