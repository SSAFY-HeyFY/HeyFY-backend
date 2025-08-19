package com.ssafy.ssashinsa.heyfy.transfer.dto;

public record CreateTransferRequest(
        String withdrawalAccountNo,
        String depositAccountNo,
        Long amount
) {}
