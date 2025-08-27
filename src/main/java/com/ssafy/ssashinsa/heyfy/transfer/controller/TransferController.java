package com.ssafy.ssashinsa.heyfy.transfer.controller;

import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferResponseDto;
import com.ssafy.ssashinsa.heyfy.transfer.docs.ForeignTransferDocs;
import com.ssafy.ssashinsa.heyfy.transfer.docs.TransferDocs;
import com.ssafy.ssashinsa.heyfy.transfer.dto.CreateTransferRequest;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferHistory;
import com.ssafy.ssashinsa.heyfy.transfer.service.TransferService;
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
    public ResponseEntity<TransferHistory> transfer(@RequestHeader("TxnAuthToken") String txnAuthToken, @RequestBody CreateTransferRequest req) {
        TransferResponseDto transferResponse = transferService.callTransfer(
                req.getDepositAccountNo(), req.getAmount(), req.getTransactionSummary(), req.getPinNumber(), txnAuthToken
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
    public ResponseEntity<TransferHistory>  foreignTransfer(@RequestHeader("TxnAuthToken") String txnAuthToken, @RequestBody CreateTransferRequest req) {
        TransferResponseDto transferResponse = transferService.callForeignTransfer(
                req.getDepositAccountNo(), req.getAmount(), req.getTransactionSummary(), req.getPinNumber(), txnAuthToken
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