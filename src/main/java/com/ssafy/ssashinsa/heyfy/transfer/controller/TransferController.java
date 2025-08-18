package com.ssafy.ssashinsa.heyfy.transfer.controller;

import com.ssafy.ssashinsa.heyfy.transfer.exception.CustomExceptions;
import com.ssafy.ssashinsa.heyfy.transfer.dto.CreateTransferRequest;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferHistory;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferHistoryResponse;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferResponseBody;
import com.ssafy.ssashinsa.heyfy.transfer.service.TransferService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.OffsetDateTime;
import java.time.ZoneId;

@RestController
@RequestMapping("/api/transfers")
@RequiredArgsConstructor
public class TransferController {

    private final TransferService transferService;

    @PostMapping
    public TransferHistoryResponse transfer(@RequestBody CreateTransferRequest req) {
        validateRequest(req);

        TransferResponseBody body = transferService.callTransfer(
                req.withdrawalAccountNo(), req.depositAccountNo(), req.amount()
        );

        if (!"H0000".equals(body.getHeader().getResponseCode())) {
            throw new CustomExceptions.InvalidRequestException(body.getHeader().getResponseMessage()); //에러 발생
        }

        var history = new TransferHistory(
                req.withdrawalAccountNo(),
                req.depositAccountNo(),
                req.amount(),
                "KRW",
                req.idempotencyKey(),
                OffsetDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return TransferHistoryResponse.ok(history);
    }

    private void validateRequest(CreateTransferRequest req) {
        if (req.amount() == null || req.amount() <= 0) {
            throw new CustomExceptions.InvalidRequestException("이체 금액은 0보다 커야 합니다.");
        }
        if (isBlank(req.withdrawalAccountNo()) || isBlank(req.depositAccountNo())) {
            throw new CustomExceptions.InvalidRequestException("출금 및 입금 계좌번호는 필수입니다.");
        }
        if (isBlank(req.idempotencyKey())) {
            throw new CustomExceptions.InvalidRequestException("요청 고유키(idempotencyKey)는 필수입니다.");
        }
    }

    private boolean isBlank(String s) {
        return s == null || s.isBlank();
    }
}