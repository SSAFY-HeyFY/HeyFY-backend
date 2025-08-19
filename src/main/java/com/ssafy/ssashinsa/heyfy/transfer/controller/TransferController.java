package com.ssafy.ssashinsa.heyfy.transfer.controller;

import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.swagger.docs.ErrorsCommonDocs;
import com.ssafy.ssashinsa.heyfy.transfer.docs.TransferDocs;
import com.ssafy.ssashinsa.heyfy.transfer.dto.*;
import com.ssafy.ssashinsa.heyfy.transfer.exception.TransferApiErrorCode;
import com.ssafy.ssashinsa.heyfy.transfer.service.TransferService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.OffsetDateTime;
import java.time.ZoneId;

@RestController
@Tag(name = "Transfer", description = "이체 관련 API")
@RequestMapping("/transfers")
@RequiredArgsConstructor
@ErrorsCommonDocs
public class TransferController {

    private final TransferService transferService;

    @PostMapping
    @TransferDocs
    public TransferHistoryResponse transfer(@RequestBody CreateTransferRequest req) {
        EntireTransferResponseDto response = transferService.callTransfer(
                req.withdrawalAccountNo(), req.depositAccountNo(), req.amount()
        );

        String responseCode = response.getHeader().getResponseCode();

        if (!"H0000".equals(responseCode)) {
            TransferApiErrorCode errorCode = TransferApiErrorCode.fromCode(responseCode);
            System.out.println("responseCode = " + responseCode);
            throw new CustomException(errorCode);
        }

        var history = new TransferHistory(
                req.withdrawalAccountNo(),
                req.depositAccountNo(),
                req.amount(),
                "KRW",
                OffsetDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return TransferHistoryResponse.ok(history);
    }
}