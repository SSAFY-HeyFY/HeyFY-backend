package com.ssafy.ssashinsa.heyfy.common.exception;

import org.springframework.http.ResponseEntity;
import org.springframework.validation.ObjectError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.Objects;

import com.ssafy.ssashinsa.heyfy.common.exception.CommonErrorCode;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(value = MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidationException(MethodArgumentNotValidException exception) {
        ObjectError objectError = Objects.requireNonNull(exception.getBindingResult().getAllErrors().stream().findFirst().orElse(null));
        return ErrorResponse.responseEntity(CommonErrorCode.INVALID_FIELD, objectError.getDefaultMessage());
    }
    // 커스텀 예외
    @ExceptionHandler(value = CustomException.class)
    public ResponseEntity<ErrorResponse> handleCustomException(CustomException e) {
        return ErrorResponse.responseEntity(e.getErrorCode());
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleAllExceptions(Exception e) {
        return ErrorResponse.responseEntity(CommonErrorCode.INTERNAL_SERVER_ERROR);
    }
}
