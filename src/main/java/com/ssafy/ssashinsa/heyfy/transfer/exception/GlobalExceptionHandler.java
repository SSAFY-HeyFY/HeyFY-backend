package com.ssafy.ssashinsa.heyfy.transfer.exception;

import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferHistoryResponse;
import com.ssafy.ssashinsa.heyfy.transfer.exception.CustomExceptions;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(CustomExceptions.InvalidRequestException.class)
    public ResponseEntity<TransferHistoryResponse> handleInvalidRequestException(CustomExceptions.InvalidRequestException ex) {
        return ResponseEntity.badRequest().body(TransferHistoryResponse.fail(ex.getMessage()));
    }

    @ExceptionHandler(CustomExceptions.ExternalApiCallException.class)
    public ResponseEntity<TransferHistoryResponse> handleExternalApiCallException(CustomExceptions.ExternalApiCallException ex) {
        return ResponseEntity.status(502).body(TransferHistoryResponse.fail(ex.getMessage()));
    }
}