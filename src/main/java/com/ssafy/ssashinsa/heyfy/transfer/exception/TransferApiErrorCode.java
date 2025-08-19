package com.ssafy.ssashinsa.heyfy.transfer.exception;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;

import java.util.Arrays;

@Getter
@RequiredArgsConstructor
public enum TransferApiErrorCode implements ErrorCode {
    // 성공 코드는 여기서 정의할 필요는 없습니다.

    // Header 관련 오류
    INVALID_HEADER_INFO("H1000", "HEADER 정보가 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    INVALID_API_NAME("H1001", "API 이름이 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    INVALID_DATE_FORMAT("H1002", "전송일자 형식이 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    INVALID_TIME_FORMAT("H1003", "전송시각 형식이 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    INVALID_INSTITUTION_CODE("H1004", "기관코드가 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    INVALID_FINTECH_SEQ_NO("H1005", "핀테크앱 일련번호가 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    INVALID_API_SERVICE_CODE("H1006", "API 서비스코드가 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    DUPLICATE_TRANSACTION_ID("H1007", "기관거래고유번호가 중복된 값입니다.", HttpStatus.CONFLICT),
    INVALID_API_KEY("H1008", "API_KEY가 유효하지 않습니다.", HttpStatus.UNAUTHORIZED),
    INVALID_USER_KEY("H1009", "USER_KEY가 유효하지 않습니다.", HttpStatus.UNAUTHORIZED),

    // 계좌 및 거래 관련 오류
    INVALID_ACCOUNT_NO("A1003", "계좌번호가 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    INVALID_TRANSACTION_AMOUNT("A1011", "거래금액이 유효하지 않습니다.", HttpStatus.BAD_REQUEST),
    INSUFFICIENT_BALANCE("A1014", "계좌잔액이 부족하여 거래가 실패했습니다.", HttpStatus.UNPROCESSABLE_ENTITY), // 처리 불가 상태(422)가 더 적절할 수 있습니다.
    TRANSFER_LIMIT_EXCEEDED_ONCE("A1016", "1회 이체가능한도를 초과했습니다.", HttpStatus.UNPROCESSABLE_ENTITY),
    TRANSFER_LIMIT_EXCEEDED_DAILY("A1017", "1일 이체가능한도를 초과했습니다.", HttpStatus.UNPROCESSABLE_ENTITY),
    TRANSACTION_SUMMARY_TOO_LONG("A1018", "거래요약내용 길이가 초과되었습니다.", HttpStatus.BAD_REQUEST),

    // 기타 요청 오류
    INVALID_JSON_FORMAT("Q1001", "요청 본문의 형식이 잘못되었습니다.", HttpStatus.BAD_REQUEST),

    // 정의되지 않은 외부 API 오류
    UNKNOWN_API_ERROR("Q1000", "알 수 없는 API 오류가 발생했습니다.", HttpStatus.INTERNAL_SERVER_ERROR);

    private final String code; // "H1000"과 같은 실제 코드를 저장할 필드
    private final String message;
    private final HttpStatus httpStatus;

    // 문자열 코드를 기반으로 Enum을 찾는 static 메서드
    public static TransferApiErrorCode fromCode(String code) {
        return Arrays.stream(values())
                .filter(errorCode -> errorCode.getCode().equals(code))
                .findFirst()
                .orElse(UNKNOWN_API_ERROR);
    }

    // ErrorCode 인터페이스의 getMessage(), getHttpStatus()는 lombok이 자동으로 구현해줍니다.
    // name()은 Enum의 기본 메서드입니다.
}
