package com.ssafy.ssashinsa.heyfy.transfer.dto;

public record TransferHistoryResponse(
        boolean success,
        TransferHistory history,
        String error // success=false일 때만 메시지
) {
    public static TransferHistoryResponse ok(TransferHistory h) {
        return new TransferHistoryResponse(true, h, null);
    }
    public static TransferHistoryResponse fail(String message) {
        return new TransferHistoryResponse(false, null, message);
    }
}
