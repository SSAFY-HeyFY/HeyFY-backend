package com.ssafy.ssashinsa.heyfy.transfer.dto;

import java.time.OffsetDateTime;

public record TransferHistory(
        String fromAccountMasked,
        String toAccountMasked,
        Long amount,
        String currency,
        String idempotencyKey,
        OffsetDateTime completedAt
) {}
