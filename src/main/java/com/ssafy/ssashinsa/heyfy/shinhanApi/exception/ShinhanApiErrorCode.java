package com.ssafy.ssashinsa.heyfy.shinhanApi.exception;

import org.springframework.http.HttpStatus;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum ShinhanApiErrorCode implements ErrorCode {
    API_CALL_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "신한API 호출 에러"),
    API_INVALID_REQUEST(HttpStatus.BAD_REQUEST, "신한API 요청 정보 오류"),
    API_USER_ALREADY_EXISTS(HttpStatus.BAD_REQUEST, "신한 은행 API에 이미 등록된 이메일입니다"); //테스트용. 후에 배포 및 서비스시 에러 메세지가 사용자에게 노출 되지 않도록 수정

    private final HttpStatus httpStatus;
    private final String message;
}
