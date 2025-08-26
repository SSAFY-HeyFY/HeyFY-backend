package com.ssafy.ssashinsa.heyfy.transfer.controller;

import com.ssafy.ssashinsa.heyfy.account.exception.AccountErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferResponseDto;
import com.ssafy.ssashinsa.heyfy.swagger.docs.ErrorsCommonDocs;
import com.ssafy.ssashinsa.heyfy.transfer.docs.ForeignTransferDocs;
import com.ssafy.ssashinsa.heyfy.transfer.docs.TransferDocs;
import com.ssafy.ssashinsa.heyfy.transfer.dto.*;
import com.ssafy.ssashinsa.heyfy.transfer.service.TransferService;
import feign.Response;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.OffsetDateTime;
import java.time.ZoneId;

@RestController
@Tag(name = "Transfer", description = "이체 관련 API")
@RequestMapping("/transfers")
@RequiredArgsConstructor
public class TransferController {

    private final TransferService transferService;

    @PostMapping("/domestic")
    @TransferDocs
    public ResponseEntity<TransferHistory> transfer(@RequestBody CreateTransferRequest req) {
        TransferResponseDto transferResponse = transferService.callTransfer(
                req.getDepositAccountNo(), req.getAmount(), req.getTransactionSummary(), req.getPinNumber()
        );

        TransferHistory history = new TransferHistory(
                req.getDepositAccountNo(),
                req.getAmount(),
                "KRW",
                req.getTransactionSummary(),
                OffsetDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return ResponseEntity.ok(history);
    }

    @PostMapping("/foreign")
    @ForeignTransferDocs
    public ResponseEntity<TransferHistory>  foreignTransfer(@RequestBody CreateTransferRequest req) {
        TransferResponseDto transferResponse = transferService.callForeignTransfer(
                req.getDepositAccountNo(), req.getAmount(), req.getTransactionSummary(), req.getPinNumber()
        );


        TransferHistory history = new TransferHistory(
                req.getDepositAccountNo(),
                req.getAmount(),
                "USD",
                req.getTransactionSummary(),
                OffsetDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return ResponseEntity.ok(history);
    }
}