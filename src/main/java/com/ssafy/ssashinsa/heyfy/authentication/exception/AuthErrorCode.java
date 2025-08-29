package com.ssafy.ssashinsa.heyfy.authentication.exception;

import org.springframework.http.HttpStatus;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum AuthErrorCode implements ErrorCode {

    UNAUTHORIZED(HttpStatus.UNAUTHORIZED, "Unauthenticated user."),
    INVALID_SIGNATURE(HttpStatus.UNAUTHORIZED, "Invalid signature."),
    INVALID_ACCESS_TOKEN(HttpStatus.UNAUTHORIZED, "Access token is invalid."),
    INVALID_REFRESH_TOKEN(HttpStatus.BAD_REQUEST, "Refresh token is invalid. Please log in again."),
    EXPIRED_TOKEN(HttpStatus.UNAUTHORIZED, "Access token has expired."),
    EXPIRED_REFRESH_TOKEN(HttpStatus.BAD_REQUEST, "Refresh token has expired. Please log in again."),
    LOGIN_FAILED(HttpStatus.BAD_REQUEST, "Invalid username or password."),
    TOKEN_PAIR_MISMATCH(HttpStatus.BAD_REQUEST, "Access token and refresh token do not match."),
    MISSING_ACCESS_TOKEN(HttpStatus.UNAUTHORIZED, "Access token is missing."),
    MISSING_REFRESH_TOKEN(HttpStatus.BAD_REQUEST, "Refresh token is missing."),
    NOT_EXPIRED_TOKEN(HttpStatus.BAD_REQUEST, "Access token is not expired."),
    EXIST_USER_NAME(HttpStatus.BAD_REQUEST, "This username is already taken."),
    EXIST_EMAIL(HttpStatus.BAD_REQUEST, "This email is already registered."),
    INVALID_PASSWORD_FORMAT(HttpStatus.BAD_REQUEST, "Password format is invalid."),
    USER_NOT_FOUND(HttpStatus.NOT_FOUND, "User not found."),
    MISSING_TXN_AUTH_TOKEN(HttpStatus.BAD_REQUEST, "Transaction authentication token is missing."),
    EXPIRED_TXN_AUTH_TOKEN(HttpStatus.BAD_REQUEST, "Transaction authentication has expired."),
    INVALID_TXN_AUTH_TOKEN(HttpStatus.BAD_REQUEST, "Transaction authentication is invalid."),
    INVALID_PIN_NUMBER(HttpStatus.BAD_REQUEST, "The secondary password you entered is incorrect."),
    TOKEN_NOT_FOUND(HttpStatus.UNAUTHORIZED, "Token not found."),
    INVALID_AUTH_HEADER(HttpStatus.UNAUTHORIZED, "Authorization header format is invalid."),
    SID_INVALID_OR_EXPIRED(HttpStatus.UNAUTHORIZED, "Session ID is invalid or has expired."),
    TOKEN_REFRESH_IN_PROGRESS(HttpStatus.CONFLICT, "Token refresh request is already in progress."),
    PIN_ATTEMPTS_EXCEEDED(HttpStatus.UNAUTHORIZED, "PIN authentication failed 5 times. Please login again."),
    PIN_TRADE_ATTEMPTS_EXCEEDED(HttpStatus.TOO_MANY_REQUESTS, "PIN authentication failed 5 times. Please try again after 30 seconds."),
    TRADE_LOCKED(HttpStatus.FORBIDDEN, "Trading is temporarily locked. Please try again after a while.");


    private final HttpStatus httpStatus;
    private final String message;
}