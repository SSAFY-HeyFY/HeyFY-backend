package com.ssafy.ssashinsa.heyfy.common.exception;

import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.exception.ShinhanException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.AuthenticationException;
import org.springframework.validation.ObjectError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.NoHandlerFoundException;

import java.util.Objects;


@Slf4j
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
        return ErrorResponse.responseEntity(e.getErrorCode(), e.getMessage());
    }

    @ExceptionHandler(value = AuthenticationException.class)
    public ResponseEntity<ErrorResponse> handleAuthenticationException(AuthenticationException e) {
        return ErrorResponse.responseEntity(AuthErrorCode.LOGIN_FAILED);
    }

    @ExceptionHandler(value = NoHandlerFoundException.class)
    public ResponseEntity<ErrorResponse> handleResourceNotFoundException(NoHandlerFoundException e) {
        return ErrorResponse.responseEntity(CommonErrorCode.RESOURCE_NOT_FOUND);
    }

    @ExceptionHandler(value = ShinhanException.class)
    public ResponseEntity<ErrorResponse> handleShinhanException(ShinhanException e) {
        log.error("Shinhan API 호출 중 예외 발생 - ErrorCode: {}, Message: {}", e.getErrorCode().name(), e.getMessage());

        ShinhanErrorCode shinhanErrorCode = e.getErrorCode();

        HttpStatus httpStatus = (shinhanErrorCode.name().startsWith("G") || shinhanErrorCode.name().startsWith("Q"))
                ? HttpStatus.INTERNAL_SERVER_ERROR // G로 시작하거나 Q로 시작하는 코드는 500
                : HttpStatus.BAD_REQUEST; // 그 외는 400

        // 후에 몇몇 메세지는 가려야 함
        String displayMessage = e.getMessage();

        ErrorResponse errorResponse = ErrorResponse.builder()
                .status(httpStatus.value())
                .httpError(httpStatus.name())
                .errorCode(shinhanErrorCode.name())
                .message(displayMessage)
                .build();

        return new ResponseEntity<>(errorResponse, httpStatus);
    }


    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleAllExceptions(Exception e) {
        return ErrorResponse.responseEntity(CommonErrorCode.INTERNAL_SERVER_ERROR);
    }



}
