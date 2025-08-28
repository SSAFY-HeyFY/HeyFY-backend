package com.ssafy.ssashinsa.heyfy.transfer.dto;

import java.time.OffsetDateTime;

public record TransferHistory(
        String depositAccountNo,
        String amount,
        String currency,
        String transactionSummary,
        OffsetDateTime completedAt,
        boolean isCorrect
) {}
