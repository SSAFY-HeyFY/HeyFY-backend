package com.ssafy.ssashinsa.heyfy.transfer.controller;

import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.transfer.TransferResponseDto;
import com.ssafy.ssashinsa.heyfy.transfer.docs.ForeignTransferDocs;
import com.ssafy.ssashinsa.heyfy.transfer.docs.TransferDocs;
import com.ssafy.ssashinsa.heyfy.transfer.dto.CreateTransferRequest;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferHistory;
import com.ssafy.ssashinsa.heyfy.transfer.service.TransferService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.OffsetDateTime;
import java.time.ZoneId;

@Slf4j
@RestController
@Tag(name = "Transfer", description = "이체 관련 API")
@RequestMapping("/transfers")
@RequiredArgsConstructor
public class TransferController {

    private final TransferService transferService;

    @PostMapping("/domestic")
    @TransferDocs
    public ResponseEntity<TransferHistory> transfer( @RequestBody CreateTransferRequest req) {
        log.info("국내 이체 요청: {}", req.getDepositAccountNo());
        try{
            TransferResponseDto transferResponse = transferService.callTransfer(
                    req.getDepositAccountNo(), req.getAmount(), req.getTransactionSummary(), req.getPinNumber()
            );
            TransferHistory history = new TransferHistory(
                    req.getDepositAccountNo(),
                    req.getAmount(),
                    "KRW",
                    req.getTransactionSummary(),
                    OffsetDateTime.now(ZoneId.of("Asia/Seoul")),
                    true
            );
            return ResponseEntity.ok(history);
        } catch (CustomException e) {
            if(e.getErrorCode().equals(AuthErrorCode.INVALID_PIN_NUMBER)){
                log.info("잘못된 핀 번호로 인한 이체 실패");
                TransferHistory history = new TransferHistory(
                        null,
                        null,
                        null,
                        null,
                        null,
                        false
                );
                return ResponseEntity.ok(history);
            }
            throw e;
        }
    }

    @PostMapping("/foreign")
    @ForeignTransferDocs
    public ResponseEntity<TransferHistory>  foreignTransfer(@RequestBody CreateTransferRequest req) {
        log.info("해외 이체 요청: {}", req.getDepositAccountNo());
        try {
            TransferResponseDto transferResponse = transferService.callForeignTransfer(
                    req.getDepositAccountNo(), req.getAmount(), req.getTransactionSummary(), req.getPinNumber()
            );


            TransferHistory history = new TransferHistory(
                    req.getDepositAccountNo(),
                    req.getAmount(),
                    "USD",
                    req.getTransactionSummary(),
                    OffsetDateTime.now(ZoneId.of("Asia/Seoul")),
                    true
            );

            return ResponseEntity.ok(history);
        }catch (CustomException e) {
            if(e.getErrorCode().equals(AuthErrorCode.INVALID_PIN_NUMBER)){
                log.info("잘못된 핀 번호로 인한 이체 실패");
                TransferHistory history = new TransferHistory(
                        null,
                        null,
                        null,
                        null,
                        null,
                        false
                );
                return ResponseEntity.ok(history);
            }
            throw e;
        }

    }
}